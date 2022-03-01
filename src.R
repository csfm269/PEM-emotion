require(glmnet)
library(e1071)
library(glmnetUtils)
library(alabama)
library(reshape2)
library(lme4)
library(tidyverse)
library(dplyr)
library(mice)
library(caret)
library(PRROC)
library(pROC)

# read data in
data.raw = read.csv("K23Data_ForBoyu_complete.csv")
ind.raw = read.csv("IndividualNAItems.csv")
data.raw$Date = as.Date(format(as.Date(data.raw$Date, format = "%m/%d/%Y"), "20%y/%m/%d"))
ind.raw$Date = as.Date(ind.raw$Date)

# within person variability
sd.in = ind.raw %>% group_by(ID) %>% summarize(Anger = sd(Angry, na.rm = T), Sad = sd(Sad, na.rm = T), Nervous = sd(Nervous, na.rm = T))
mean(sd.in$Anger)
mean(sd.in$Sad)
mean(sd.in$Nervous)

data.merge = left_join(data.raw, ind.raw, by = c("ID","Date","Hour"))
data.merge = data.merge %>% mutate(GPS_freq = case_when(is.na(GPS_freq)~0,
                                                        ceiling(GPS_freq)!=GPS_freq ~ 1,
                                                        GPS_freq==3824 ~ 1,
                                                        TRUE ~ GPS_freq))


data.merge$ID = as.character(factor(data.merge$ID, labels = c(paste("TD", 1:10, sep = ""), paste("AH", 1:13, sep = ""))))
data.ang = data.merge[!is.na(data.merge$Angry),] %>% 
  dplyr::select(ID = -c(NA..x, PA.x, NA..y, PA.y, X, Ruminate, K23.Week, PriorStressor, Sad, Nervous, GPS_dist))
data.sad = data.merge[!is.na(data.merge$Sad),] %>% 
  dplyr::select(ID = -c(NA..x, PA.x, NA..y, PA.y, X, Ruminate, K23.Week, PriorStressor, Angry, Nervous, GPS_dist))
data.nev = data.merge[!is.na(data.merge$Nervous),] %>% 
  dplyr::select(ID = -c(NA..x, PA.x, NA..y, PA.y, X, Ruminate, K23.Week, PriorStressor, Angry, Sad, GPS_dist))

#preparing data
data.ang.imp = mice(data.ang%>%dplyr::select(-c(Angry,ID)))
data.ang.complete = complete(data.ang.imp)
data.ang.complete$Angry = data.ang$Angry
data.ang.complete$ID = data.ang$ID
data.ang.use = lapply(split(data.ang.complete, data.ang.complete$ID), function(x){
  x$ang.rescale = x$Angry - median(x$Angry)
  x$ang.extreme = 1*(x$ang.rescale>0.5)#*(c(0, 1*(diff(x$Angry)>1)))
  x
})

data.sad.imp = mice(data.sad%>%dplyr::select(ID=-Sad))
data.sad.complete = complete(data.sad.imp)
data.sad.complete$Sad = data.sad$Sad
data.sad.use = lapply(split(data.sad.complete, data.sad.complete$ID), function(x){
  x$sad.rescale = x$Sad - median(x$Sad)
  x$sad.extreme = 1*(x$sad.rescale>0.5)
  x
})

data.nerv.imp = mice(data.nev%>%dplyr::select(ID=-Nervous))
data.nerv.complete = complete(data.nerv.imp)
data.nerv.complete$Nervous = data.nev$Nervous
data.nerv.use = lapply(split(data.nerv.complete, data.nerv.complete$ID), function(x){
  x$nev.rescale = x$Nervous - median(x$Nervous)
  x$nev.extreme = 1*(x$nev.rescale>0.5)#*(c(0, 1*(diff(x$nevry)>1)))
  x
})

# some statistics
data.ang.filter = data.ang.use[sapply(data.ang.use, function(x) sum(x$ang.extreme)>4)]
total.n = rowSums(sapply(data.ang.filter, function(x) c(sum(x$ang.extreme), nrow(x))))
total.n[1]/total.n[2]
mean( sapply(data.ang.filter, function(x) mean(x$Angry[x$ang.extreme==1] - mean(x$Angry))) )
mean(sapply(data.ang.filter, function(x) sd(x$Angry)))

data.sad.filter = data.sad.use[sapply(data.sad.use, function(x) sum(x$sad.extreme)>4)]
total.n = rowSums(sapply(data.sad.filter, function(x) c(sum(x$sad.extreme), nrow(x))))
total.n[1]/total.n[2]
mean( sapply(data.sad.filter, function(x) mean(x$Sad[x$sad.extreme==1] - mean(x$Sad))) )
mean(sapply(data.sad.filter, function(x) sd(x$Sad)))

data.nerv.filter = data.nerv.use[sapply(data.nerv.use, function(x) sum(x$nev.extreme)>4)]
total.n = rowSums(sapply(data.nerv.filter, function(x) c(sum(x$nev.extreme), nrow(x))))
total.n[1]/total.n[2]
mean( sapply(data.nerv.filter, function(x) mean(x$Nervous[x$nev.extreme==1] - mean(x$Nervous))) )
mean(sapply(data.nerv.filter, function(x) sd(x$Nervous)))

logistic.loss = function(y, p.pred){
  n.pos = sum(y==1)
  n.neg = sum(y==0)
  p.pred[p.pred<=0] = 1e-3
  p.pred[p.pred>=1] = 1 - 1e-3
  n.pos = sum(y==1)
  -mean(y*log(p.pred) + (1-y)*log(1-p.pred))
}

w.simplex = function(p.stacking, y, L, lambda){
  # browser()
  K = ncol(p.stacking)/L
  target.fn = function(x){
    # try to maximize C-statistics
    p.pred = p.stacking%*%x
    # p.pos = p.pred[y==1]
    # p.neg = p.pred[y==0]
    # n.pos = sum(y==1)
    # n.neg = sum(y==0)
    # diff = outer(p.pos, p.neg, "-")
    # sum(-((diff>gamma)*(diff-gamma))^2)/length(p.pos)/length(p.neg)
    # sum(log(1+exp(-3*diff)))/length(p.pos)/length(p.neg)
    # sum((1-diff)^2)/length(p.pos)/length(p.neg)
    # sum((y - p.pred)^2)
    logistic.loss(y, p.pred) #+ lambda*sum((x - w.gen.new)^2)
  }
  res = auglag(c(rep(0.9/L,L), rep(0.1/(K-1)/L,(K-1)*L)),
         target.fn,
         hin = function(x) x,
         heq = function(x) sum(x)-1, 
         control.outer = list(trace = F))
  res$par
}

# selecting lambda
# res.all = sapply(seq(0,1,0.2)^2, function(alpha){
#   cv.res = cv.glmnet(x = X, y = y, family = "binomial", alpha = alpha)
#   c(min(cv.res$cvm), cv.res$lambda.min)
# })
# alpha.opt = (seq(0,1,0.1)^3)[order(res.all[1,])[1]]
# lambda.opt = res.all[2,order(res.all[1,])[1]]
# cv.res = cv.glmnet( x = X, y = y, family = "binomial", alpha = 0.1)
# glmnet(x = X, y = y, family = "binomial", alpha = 0.1, lambda = cv.res$lambda.min)

enet.wrapper = function(X, y){
  if(sum(y==1)<=3){
    y.tmp = sample(c(1,1,rep(0, length(y)-2)))
    glm(y~., family = "binomial", data = data.frame(y = y.tmp, X))
  }
  glm(y~., family = "binomial", data = data.frame(y = y, X))
}

svm.wrapper = function(X, y){
  svm(x = X, y = y, type = "C-classification", probability = T)
}

rf.wrapper = function(X, y){
  randomForest::randomForest(x = X, y = y)
}

specialist.wrapper = function(X, y, learner.lst, other.models, lambda = 0, nfolds = 10){
  # browser()
  
  # learner can be multiple objects
  # if(nrow(X)<=10){
  #   folds = createFolds(1:nrow(X), nrow(X))
  # }else{
  #   folds = createFolds(1:nrow(X), 10)
  # }
  # folds = createFolds(1:nrow(X), nrow(X))
  folds = createFolds(y, k = nfolds)

  L = length(learner.lst)
  
  # create 
  p.stacking = do.call(rbind, lapply(folds, function(fold){
    in.models = lapply( learner.lst, function(learner){
      learner(X[-fold,,drop = F], y[-fold])
    })
    all.models = c(in.models, other.models)
    
    sapply(all.models, function(ind.model){
      if(class(ind.model)[1] == "glm"){
        predict(ind.model, data.frame(X[fold,,drop = F]), type = "response")
      }else if(class(ind.model)[1] == "svm"){
        attributes(predict(ind.model, X[fold,,drop = F], probability = T))$probabilities[,2]
      }else if(class(ind.model)[1] == "randomForest"){
        predict(ind.model, X[fold,,drop = F], type = "prob")[,2]
      }
    })
  }))
  
  y.obs = y[unlist(folds)]
  # cv.res = cv.glmnet(x = p.stacking, y = y.obs, family = "binomial", alpha = 1)
  # glm.res = glmnet(x = p.stacking, y = y.obs, family = "binomial", alpha = 1, lambda = cv.res$lambda.min)
  # c(glm.res$a0, as.numeric(glm.res$beta))
  # nnls(A = p.stacking, b = as.numeric(y.obs)-1)$x
  w.simplex( p.stacking, as.numeric(y.obs) - 1, L, lambda = lambda)
}

penalized.stack.wrapper = function(data.ls, formula, outcome, learner.lst){
  # browser()
  # L is the number of algorithms in consideration
  L = length(learner.lst)
  all.models = do.call(c, lapply(data.ls, function(data.ind){
    # get rid of the intercept
    X = model.matrix(formula, data = data.ind)[,-1]
    y = as.factor(data.ind[[outcome]])
    model.res = lapply(learner.lst, function(learner){
      learner(X, y)
    })
    model.res
  }))

  # specialist training
  specialist.auc.all = lapply(seq_along(data.ls), function(idx){
    print(idx)
    X = model.matrix(formula, data = data.ls[[idx]])[,-1]
    y = as.factor(data.ls[[idx]][[outcome]])

    folds = createFolds(y, k = 10)
    remove.idx = (idx-1)*L + (1:L)

    res.stack = lapply(folds, function(fold){
      print(fold)
      in.models = lapply(learner.lst, function(learner){
        learner(X[-fold,,drop=F], y[-fold])
      })
      w.spe = specialist.wrapper(X[-fold,,drop=F], y[-fold], learner.lst, all.models[-remove.idx], nfolds = 10)

      all.pred = sapply(c(in.models, all.models[-remove.idx]), function(ind.model){
        if(class(ind.model)[1] == "glm"){
          predict(ind.model, data.frame(X[fold,,drop = F]), type = "response")
        }else if(class(ind.model)[1] == "svm"){
          attributes(predict(ind.model, X[fold,,drop = F], probability = T))$probabilities[,2]
        }else if(class(ind.model)[1] == "randomForest"){
          predict(ind.model, X[fold,,drop = F], type = "prob")[,2]
        }
      })
      if(is.null(dim(all.pred))){
        p.in = matrix(all.pred[1:L], nrow = 1)
      }else{
        p.in = all.pred[,1:L,drop = F]
      }
      list( p.pred = all.pred%*%w.spe, w = w.spe, p.in = p.in)
    })

    p.stack = unlist(lapply(res.stack, function(x) x$p.pred))
    p.in = do.call(rbind, lapply(res.stack, function(x) x$p.in))
    w.all = sapply(res.stack, function(x) x$w)
    y.obs = y[unlist(folds)]

    list(data = data.frame(ID = names(data.ls)[idx], day = data.ls[[idx]]$day[unlist(folds)], 
                           y = y.obs, p.stack = p.stack, p.in = p.in), w = rowMeans(w.all))
  })
  all.pred = do.call(rbind, lapply(specialist.auc.all, function(x) x$data))

  w.all = sapply(seq_along(specialist.auc.all), function(i){
    tmp = specialist.auc.all[[i]]$w
    weight = tmp
    weight[i] = tmp[1]
    weight[-i] = tmp[-1]
    weight
  })
  colnames(w.all) = names(data.ls)
  return(list(res = all.pred, w = w.all, ind = specialist.auc.all))
}

analysis.in = function(outcome, data.use, learner, loss = c("AUC", "MSE")){
  formula.pca = as.formula("~ActScore + GPS_freq + ActScore + missing + homDist+
    radiusMobility + percentHome + numPlaces + Power +
    SleepOnset_time + Wake_time + Sleep_Duration +
    mphuse + hphuse + TimeDay")
  df.ls = pca.preproc(data.use, outcome, formula.pca)
  formula.data = as.formula(paste(outcome,".-ID-day",sep = "~"))
  data.ls = df.ls[sapply(df.ls, function(x) sum(x[[outcome]]))>4]
  
  sapply(data.ls, function(data.ind){
    folds = createFolds(data.ind[[outcome]], k = 5)
    X = model.matrix(formula.data, data = data.ind)[,-1]
    y = as.factor(data.ind[[outcome]])
    
    p.in = unlist(lapply(folds, function(fold){
      ind.model = learner(X[-fold,,drop=F], y[-fold])
      if(class(ind.model)[1] == "glm"){
        predict(ind.model, data.frame(X[fold,,drop = F]), type = "response")
      }else if(class(ind.model)[1] == "svm"){
        attributes(predict(ind.model, X[fold,,drop = F], probability = T))$probabilities[,2]
      }else if(class(ind.model)[1] == "randomForest"){
        predict(ind.model, X[fold,,drop = F], type = "prob")[,2]
      }
      }))
    if(loss == "AUC"){
      auc(y[unlist(folds)], p.in, direction = "<")
    }else if(loss == "MSE"){
      mean( (as.numeric(y[unlist(folds)])-1 - p.in)^2 )
    }
    
    })
}

penalized.stack.tsCV.wrapper = function(data.ls, formula, outcome, learner.lst, test.prop = 0.75){
  data.ls = lapply(data.ls, function(data.ind){
    data.ind[order(data.ind$day),]
  })
  
  # L is the number of algorithms in consideration
  L = length(learner.lst)
  all.models = do.call(c, lapply(data.ls, function(data.ind){
    # get rid of the intercept
    X = model.matrix(formula, data = data.ind)[,-1]
    y = as.factor(data.ind[[outcome]])
    model.res = lapply(learner.lst, function(learner){
      learner(X, y)
    })
    model.res
  }))
  
  # specialist training
  specialist.auc.all = lapply(seq_along(data.ls), function(idx){
    print(idx)
    X = model.matrix(formula, data = data.ls[[idx]])[,-1]
    y = as.factor(data.ls[[idx]][[outcome]])
    
    n.all = nrow(X)
    test.idx = (round(test.prop*n.all)):n.all
    remove.idx = (idx-1)*L + (1:L)
    
    res.stack = lapply(test.idx, function(test.ind){
      train.idx = 1:(test.ind-1)
      in.models = lapply(learner.lst, function(learner){
        learner(X[train.idx,,drop=F], y[train.idx])
      })
      w.spe = specialist.wrapper(X[train.idx,,drop=F], y[train.idx], learner.lst, all.models[-remove.idx], nfolds = 10)
      
      all.pred = sapply(c(in.models, all.models[-remove.idx]), function(ind.model){
        if(class(ind.model)[1] == "glm"){
          predict(ind.model, data.frame(X[test.ind,,drop = F]), type = "response")
        }else if(class(ind.model)[1] == "svm"){
          attributes(predict(ind.model, X[test.ind,,drop = F], probability = T))$probabilities[,2]
        }else if(class(ind.model)[1] == "randomForest"){
          predict(ind.model, X[test.ind,,drop = F], type = "prob")[,2]
        }
      })
      if(is.null(dim(all.pred))){
        p.in = matrix(all.pred[1:L], nrow = 1)
      }else{
        p.in = all.pred[,1:L,drop = F]
      }
      list( p.pred = all.pred%*%w.spe, w = w.spe, p.in = p.in)
    })
    
    p.stack = unlist(lapply(res.stack, function(x) x$p.pred))
    p.in = do.call(rbind, lapply(res.stack, function(x) x$p.in))
    w.all = sapply(res.stack, function(x) x$w)
    y.obs = y[test.idx]
    
    list(data = data.frame(ID = names(data.ls)[idx], day = data.ls[[idx]]$day[test.idx], 
                           y = y.obs, p.stack = p.stack, p.in = p.in), w = rowMeans(w.all))
  })
  all.pred = do.call(rbind, lapply(specialist.auc.all, function(x) x$data))
  all.pred
}

glmer.wrapper = function(data.use, outcome){
  all.folds = lapply(data.use, function(x) createFolds(x[[outcome]], 10))
  vars = colnames(data.use[[1]])
  vars = vars[-c(length(vars)-2, length(vars)-1, length(vars))]
  glmer.formula = as.formula( sprintf( "%s~(1|ID) + %s", outcome, paste(vars, collapse = "+") ) )
  glmer.res = do.call(rbind, lapply(1:10, function(fold){
    print(fold)
    data.train = do.call(rbind, lapply(seq_along(data.use), function(study){
      fold.idx = all.folds[[study]][[fold]]
      data.use[[study]][-fold.idx,]
    }))
    data.test = do.call(rbind, lapply(seq_along(data.use), function(study){
      fold.idx = all.folds[[study]][[fold]]
      data.use[[study]][fold.idx,]
    }))
    
    glmer.res = glmer(glmer.formula, data = data.train, family = "binomial")
    p.glmer = predict(glmer.res, data.test, type = "response")
    y = data.test[[outcome]]
    data.frame(y = y, p.glmer = p.glmer)
  }))
  glmer.res
}

pca.preproc = function(data.use, outcome, formula.pca){
  data.all = do.call(rbind, data.use)
  data.all.mt = model.matrix(formula.pca, data.all)[,-1]
  data.pca = princomp(data.all.mt)
  # n.pcs = sum(cumsum(data.pca$sdev^2/sum(data.pca$sdev^2))<0.9) + 1
  n.pcs = 5
  X.data.pca = data.pca$scores[,1:n.pcs]
  df.tmp = data.frame(X.data.pca, ID = data.all$ID, day = data.all$day, data.all[[outcome]])
  colnames(df.tmp)[ncol(df.tmp)] = outcome
  return( split(df.tmp, df.tmp$ID) )
}

get.import = function(data.complete, name){
  # browser()
  formula.var = "ActScore + GPS_freq + ActScore + missing + homDist+
    radiusMobility + percentHome + numPlaces + Power +
    SleepOnset_time + Wake_time + Sleep_Duration +
    mphuse + hphuse + TimeDay"
  var.extreme = paste(name, "extreme", sep = ".")
  formula.use = as.formula(paste(var.extreme, formula.var, sep = "~"))
  data.complete[[var.extreme]] = as.factor(data.complete[[var.extreme]])
  data.complete$TimeDay = as.numeric(factor(data.complete$TimeDay, levels = c("Morning", "Afternoon", "Evening/Night"), ordered = T))
  data.complete.split = split(data.complete, data.complete$ID)
  ind.import = lapply(data.complete.split, function(x){
    if(sum(x[[var.extreme]]==1)==0){
      return(NA)
    }else{
      rf.res = randomForest::randomForest(formula.use, data = x)
      rf.sign = sapply( row.names(rf.res$importance), function(pred.var){
        cor.partial = cor(partial.bin(rf.res, x, pred.var))[1,2]
        sign.partial = sign(cor.partial)
        sign.partial[is.na(sign.partial)] = 1
        sign.partial
      })
      return(rf.res$importance/sum(rf.res$importance)*rf.sign)
    }
  })
  do.call(cbind, ind.import)
}

analysis.wrapper = function(outcome, data.use, cv = c("regular", "ts")){
  # browser()
  formula.pca = as.formula("~ActScore + GPS_freq + ActScore + missing + homDist+
    radiusMobility + percentHome + numPlaces + Power +
    SleepOnset_time + Wake_time + Sleep_Duration +
    mphuse + hphuse + TimeDay")
  df.ls = pca.preproc(data.use, outcome, formula.pca)
  formula.data = as.formula(paste(outcome,".-ID-day",sep = "~"))
  data.ls = df.ls[sapply(df.ls, function(x) sum(x[[outcome]]))>4]
  
  if(cv == "regular"){
    glmer.res = glmer.wrapper(data.ls, outcome)
    enet.res = penalized.stack.wrapper(data.ls, formula.data, outcome, c(enet.wrapper))
    svm.res = penalized.stack.wrapper(data.ls, formula.data, outcome, c(svm.wrapper))
    rf.res = penalized.stack.wrapper(data.ls, formula.data, outcome, c(rf.wrapper))
    de.res = penalized.stack.wrapper(data.ls, formula.data, outcome, c(enet.wrapper, svm.wrapper, rf.wrapper))
  }else if(cv == "ts"){
    glmer.res = NA
    enet.res = penalized.stack.tsCV.wrapper(data.ls, formula.data, outcome, c(enet.wrapper))
    svm.res = penalized.stack.tsCV.wrapper(data.ls, formula.data, outcome, c(svm.wrapper))
    rf.res = penalized.stack.tsCV.wrapper(data.ls, formula.data, outcome, c(rf.wrapper))
    de.res = penalized.stack.tsCV.wrapper(data.ls, formula.data, outcome, c(enet.wrapper, svm.wrapper, rf.wrapper))
  }
  
  list(enet.res, svm.res, rf.res, de.res, glmer.res)
}

pred.traj.plot = function(res){
  res.roc = roc(res$y, res$p.stack, direction = "<")
  opt.thresh = res.roc$thresholds[order((1-res.roc$sensitivities)^2 + (1-res.roc$specificities)^2)[1]]
  res$y = as.numeric(res$y) - 1
  y.pred = 1*(res$p.stack>opt.thresh)
  res$correct = (res$y == y.pred)
  
  res.parsed = res %>% group_by(ID, day) %>% summarise( y = max(y), correct = max(correct) )
  ggplot(data = res.parsed, aes(x = day, y = ID, fill = as.factor(y))) + geom_tile() +
    geom_point(aes(shape = ifelse(correct==1, "correct", "incorrect")), show.legend = F) +
    scale_shape_manual(values = c(incorrect = 4, correct = NA), guide = "none") + 
    coord_fixed() + scale_fill_manual(values = c("#00BFC4", "#F8766D"), labels = c("Non-HNA", "HNA"), 
                                      name = "Actual\nstate") +
    theme_bw() + xlab("Days since enrollment") + ylab("")
}

summarize.analysis = function( all.res ){
  # get auc
  auc.merge = lapply(seq_along(all.res), function(i){
    if(i==5){
      roc(all.res[[i]]$y, all.res[[i]]$p.glmer)
    }else if(i==4){
      roc(all.res[[i]]$res$y, all.res[[i]]$res$p.stack)
    }else{
      list(roc(all.res[[i]]$res$y, all.res[[i]]$res$p.stack), 
        roc(all.res[[i]]$res$y, all.res[[i]]$res$V1))
    }
  })
  auc.ind = lapply(seq_along(all.res), function(i){
    if(i<4){
      sapply(all.res[[i]]$ind, function(x){
        c( auc(x$data$y, x$data$p.stack, direction = "<"),
           auc(x$data$y, x$data$V1, direction = "<") )
      })
    }else if(i==4){
      sapply(all.res[[i]]$ind, function(x){
        auc(x$data$y, x$data$p.stack, direction = "<")
      })
    }
  })
  list(merge = auc.merge, ind = auc.ind)
}

plot.auc = function(all.auc){
  all.length = c(length(all.auc$merge[[1]][[1]]$sensitivities), length(all.auc$merge[[2]][[1]]$sensitivities),
                 length(all.auc$merge[[3]][[1]]$sensitivities), length(all.auc$merge[[4]]$sensitivities),
                 length(all.auc$merge[[5]]$sensitivities))
  all.sen = c(all.auc$merge[[1]][[1]]$sensitivities, all.auc$merge[[2]][[1]]$sensitivities, all.auc$merge[[3]][[1]]$sensitivities,
              all.auc$merge[[4]]$sensitivities, all.auc$merge[[5]]$sensitivities)
  all.spe = c(all.auc$merge[[1]][[1]]$specificities, all.auc$merge[[2]][[1]]$specificities, all.auc$merge[[3]][[1]]$specificities,
              all.auc$merge[[4]]$specificities, all.auc$merge[[5]]$specificities)
  plt.df = data.frame(sensitivity = all.sen, specificity = all.spe, 
                      Method = factor(rep(c("PEM-ENet", "PEM-SVM", "PEM-RF", "PDEM", "GLMER"), all.length), 
                                      levels = c("GLMER", "PEM-ENet", "PEM-SVM", "PEM-RF", "PDEM"),
                                      ordered = T))
  ggplot(plt.df, aes(x = 1-specificity, y = sensitivity, color = Method)) + geom_line() + scale_color_discrete()+
    theme_bw() + geom_abline(slope = 1, intercept = 0, linetype = 2) + coord_fixed() + xlab("1 - Specificity") + ylab("Sensitivity") +
    theme(axis.text = element_text(size = 16), 
          axis.title = (element_text(size = 18)), plot.title = element_text(size = 20),
          legend.text = element_text(size = 16), legend.title = element_text(size = 18))
}

get.optim.accuracy = function( y, p.pred ){
  p.prop = mean(y==1)
  roc.res = roc(y, p.pred)
  opt.threshold = roc.res$thresholds[order((1 - roc.res$sensitivities)^2 + (1 - roc.res$specificities)^2)[1]]
  y.pred = 1*(p.pred>=opt.threshold)
  mean(y==y.pred)
}

table.auc = function(all.auc){
  c( auc(all.auc$merge[[1]][[1]]), auc(all.auc$merge[[2]][[1]]), auc(all.auc$merge[[3]][[1]]),
     auc(all.auc$merge[[4]]), auc(all.auc$merge[[5]]))
}

table.acc = function(all.res){
  c(sapply(all.res[1:4], function(x){
    get.optim.accuracy(x$res$y, x$res$p.stack)
  }), get.optim.accuracy(all.res[[5]]$y, all.res[[5]]$p.glmer))
}

# regular CV prediction
ang.all.res = analysis.wrapper("ang.extreme", data.ang.use, cv = "regular")
sad.all.res = analysis.wrapper("sad.extreme", data.sad.use, cv = "regular")
nerv.all.res = analysis.wrapper("nev.extreme", data.nerv.use, cv = "regular")

# ROC plots
plot.auc(summarize.analysis(ang.all.res))
plot.auc(summarize.analysis(sad.all.res))
plot.auc(summarize.analysis(nerv.all.res))

table.auc(summarize.analysis(ang.all.res))
table.auc(summarize.analysis(sad.all.res))
table.auc(summarize.analysis(nerv.all.res))

# accuracy
table.acc(ang.all.res)
table.acc(sad.all.res)
table.acc(nerv.all.res)

# ind difference
all.methods = c(enet.wrapper, svm.wrapper, rf.wrapper)
all.methods.names = c("PEM-ENet", "PEM-SVM", "PEM-RF")
auc.in.res = do.call(rbind, lapply(1:3, function(idx.method){
  ang.in = analysis.in("ang.extreme", data.ang.use, all.methods[[idx.method]], loss = "AUC")
  sad.in = analysis.in("sad.extreme", data.sad.use, all.methods[[idx.method]], loss = "AUC")
  nerv.in = analysis.in("nev.extreme", data.nerv.use, all.methods[[idx.method]], loss = "AUC")
  data.frame(acc = c(ang.in, sad.in, nerv.in),
             ID = c(1:length(ang.in), 1:length(sad.in), 1:length(nerv.in)),
             Emotion = rep(c("Anger", "Sadness", "Nervousness"), c(length(ang.in),length(sad.in),length(nerv.in))),
             Method = all.methods.names[idx.method],
             Metric = "AUC")
}))
mse.in.res = do.call(rbind, lapply(1:3, function(idx.method){
  ang.in = analysis.in("ang.extreme", data.ang.use, all.methods[[idx.method]], loss = "MSE")
  sad.in = analysis.in("sad.extreme", data.sad.use, all.methods[[idx.method]], loss = "MSE")
  nerv.in = analysis.in("nev.extreme", data.nerv.use, all.methods[[idx.method]], loss = "MSE")
  data.frame(acc = -c(ang.in, sad.in, nerv.in),
             Emotion = rep(c("Anger", "Sadness", "Nervousness"), c(length(ang.in),length(sad.in),length(nerv.in))),
             ID = c(1:length(ang.in), 1:length(sad.in), 1:length(nerv.in)),
             Method = all.methods.names[idx.method],
             Metric = "Brier score")
}))

all.res = list(ang.all.res, sad.all.res, nerv.all.res)
all.emotion = c("Anger", "Sadness", "Nervousness")
auc.stack.res = do.call(rbind, lapply(1:3, function(ind.idx){
  enet.stack = sapply(all.res[[ind.idx]][[1]]$ind, function(x){
    auc(x$data$y, x$data$p.stack, direction = "<")
  })
  svm.stack = sapply(all.res[[ind.idx]][[2]]$ind, function(x){
    auc(x$data$y, x$data$p.stack, direction = "<")
  })
  rf.stack = sapply(all.res[[ind.idx]][[3]]$ind, function(x){
    auc(x$data$y, x$data$p.stack, direction = "<")
  })
  
  data.frame(acc = c(enet.stack, svm.stack, rf.stack), 
             Emotion = all.emotion[ind.idx],
             ID = rep(1:length(enet.stack), 3),
             Method = rep(c("PEM-ENet", "PEM-SVM", "PEM-RF"), each = length(enet.stack)),
             Metric = "AUC")
}))

mse.stack.res = do.call(rbind, lapply(1:3, function(ind.idx){
  enet.stack = sapply(all.res[[ind.idx]][[1]]$ind, function(x){
    mean( (as.numeric(x$data$y) - 1 - x$data$p.stack)^2 )
  })
  svm.stack = sapply(all.res[[ind.idx]][[2]]$ind, function(x){
    mean( (as.numeric(x$data$y) - 1 - x$data$p.stack)^2 )
  })
  rf.stack = sapply(all.res[[ind.idx]][[3]]$ind, function(x){
    mean( (as.numeric(x$data$y) - 1 - x$data$p.stack)^2 )
  })
  
  data.frame(acc = -c(enet.stack, svm.stack, rf.stack), 
             Emotion = all.emotion[ind.idx],
             ID = rep(1:length(enet.stack), 3),
             Method = rep(c("PEM-ENet", "PEM-SVM", "PEM-RF"), each = length(enet.stack)),
             Metric = "Brier score")
}))

auc.res = left_join(auc.in.res, auc.stack.res, by = c("Emotion", "Method", "Metric", "ID"))
mse.res = left_join(mse.in.res, mse.stack.res, by = c("Emotion", "Method", "Metric", "ID"))
auc.res$Emotion = factor(auc.res$Emotion, levels = c("Anger", "Sadness", "Nervousness"), ordered = T)
mse.res$Emotion = factor(mse.res$Emotion, levels = c("Anger", "Sadness", "Nervousness"), ordered = T)
auc.res$diff = auc.res$acc.y - auc.res$acc.x
mse.res$diff = -log(mse.res$acc.y/mse.res$acc.x)

auc.diff.fig = ggplot(auc.res, aes(x = Emotion, y = diff, fill = Method)) + geom_boxplot(outlier.shape = NA) + geom_hline(yintercept = 0, linetype = 2) +
  theme_bw() + ylab("AUC(PEM) - AUC(IM)") + ylim(c(-0.3,0.3)) + scale_fill_discrete()
mse.diff.fig = ggplot(mse.res, aes(x = Emotion, y = diff, fill = Method)) + geom_boxplot(outlier.shape = NA) + geom_hline(yintercept = 0, linetype = 2) +
  theme_bw() + ylab("log(BS(IM)/BS(PEM))") + ylim(c(-0.3,0.3)) + scale_fill_discrete()


ggsave("./AUC_res/ind_auc_diff.pdf", auc.diff.fig + ggtitle("AUC difference"), width = 6, height = 5)
ggsave("./AUC_res/ind_mse_diff.pdf", mse.diff.fig + ggtitle("Brier score difference"), width = 6, height = 5)

#### decision curves ####
calc.decision = function(y, p.pred){
  sapply(seq(0,0.99,0.01), function(threshold){
    y.pred = (p.pred>threshold)*1
    tpr = sum(y.pred*y)/length(y)
    fpr = sum(y.pred*(1-y))/length(y)
    tpr - fpr*threshold/(1-threshold)
  })
}

plot.decision = function(all.res){
  dec.enet = calc.decision(as.numeric(all.res[[1]]$res$y)-1, all.res[[1]]$res$p.stack)
  dec.svm = calc.decision(as.numeric(all.res[[2]]$res$y)-1, all.res[[2]]$res$p.stack)
  dec.rf = calc.decision(as.numeric(all.res[[3]]$res$y)-1, all.res[[3]]$res$p.stack)
  dec.pdem = calc.decision(as.numeric(all.res[[4]]$res$y)-1, all.res[[4]]$res$p.stack)
  dec.glmm = calc.decision(all.res[[5]]$y, all.res[[5]]$p.glmer)
  dec.always = calc.decision(all.res[[5]]$y, rep(1,nrow(all.res[[5]])))
  plt.dec = data.frame(benefit = c(dec.enet, dec.svm, dec.rf, dec.pdem, dec.glmm),
                       threshold = rep(seq(0,0.99,0.01), 5),
                       method = factor(rep(c("PEM-Enet", "PEM-SVM", "PEM-RF", "PDEM", "GLMER"), each = 100), 
                                          level = c("GLMER","PEM-Enet", "PEM-SVM", "PEM-RF", "PDEM"), ordered = T))
  ggplot() + geom_line(data = plt.dec, aes(x = threshold, y = benefit, color = method)) + scale_color_discrete(name = "Method") +
    ylim(c(-0.05,0.3)) + geom_hline(yintercept = 0, linetype = 2) + theme_bw() + xlab("Threshold") + ylab("Benefit") +
    geom_line(data = data.frame(x = seq(0,0.99,0.01), y = dec.always), aes(x = x, y = y)) +
    annotate("text", x = 0.1, y = 0.1, label = "Always\nintervene")
}

ggsave("AUC_res/ang_decision_curve_updated.pdf", plot.decision(ang.all.res) + ggtitle("Anger"), width = 5.5, height = 4 )
ggsave("AUC_res/sad_decision_curve_updated.pdf", plot.decision(sad.all.res) + ggtitle("Sadness"), width = 5.5, height = 4 )
ggsave("AUC_res/nerv_decision_curve_updated.pdf", plot.decision(nerv.all.res) + ggtitle("Nervousness"), width = 5.5, height = 4 )

#### stacking weights ####
ang.w = ang.all.res[[3]]$w
row.names(ang.w) = colnames(ang.w)
pdf("ind_weights/Anger_weights.pdf", width = 9, height = 8)
ggplot(data = melt(ang.w), aes(x = Var1, y = Var2, fill = value)) + geom_tile(color = "black") + 
  scale_fill_gradient2(low = "blue", high = "red", name = "Stacking\nweight", guide = guide_colourbar(barheight = 18)) +
  scale_x_discrete(expand = c(0,0)) + scale_y_discrete(expand = c(0,0)) + coord_fixed() +
  xlab("Idiographic models") + ylab("PEM") + ggtitle("Anger") + geom_abline(slope = 1, intercept = 0, linetype = 2, size = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text = element_text(size = 16), 
        axis.title = (element_text(size = 18)), plot.title = element_text(size = 20),
        legend.text = element_text(size = 16), legend.title = element_text(size = 18))
dev.off()

sad.w = sad.all.res[[3]]$w
row.names(sad.w) = colnames(sad.w)
pdf("ind_weights/Sad_weights.pdf", width = 9, height = 8)
ggplot(data = melt(sad.w), aes(x = Var1, y = Var2, fill = value)) + geom_tile(color = "black") + 
  scale_fill_gradient2(low = "blue", high = "red", name = "Stacking\nweight", guide = guide_colourbar(barheight = 18)) +
  scale_x_discrete(expand = c(0,0)) + scale_y_discrete(expand = c(0,0)) + coord_fixed() +
  xlab("Idiographic models") + ylab("PEM") + ggtitle("Sadness") + geom_abline(slope = 1, intercept = 0, linetype = 2, size = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text = element_text(size = 16), 
        axis.title = (element_text(size = 18)), plot.title = element_text(size = 20),
        legend.text = element_text(size = 16), legend.title = element_text(size = 18))
dev.off()

nerv.w = nerv.all.res[[3]]$w
row.names(nerv.w) = colnames(nerv.w)
pdf("ind_weights/Nerv_weights.pdf", width = 9, height = 8)
ggplot(data = melt(nerv.w), aes(x = Var1, y = Var2, fill = value)) + geom_tile(color = "black") + 
  scale_fill_gradient2(low = "blue", high = "red", name = "Stacking\nweight", guide = guide_colourbar(barheight = 18)) +
  scale_x_discrete(expand = c(0,0)) + scale_y_discrete(expand = c(0,0)) + coord_fixed() +
  xlab("Idiographic models") + ylab("PEM") + ggtitle("Anger") + geom_abline(slope = 1, intercept = 0, linetype = 2, size = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text = element_text(size = 16), 
        axis.title = (element_text(size = 18)), plot.title = element_text(size = 20),
        legend.text = element_text(size = 16), legend.title = element_text(size = 18))
dev.off()

#### predicted trajectories ####
ggsave("./prediction_traj/ang_traj.pdf", pred.traj.plot(ang.all.res[[3]]$res) + ggtitle("Anger"), width = 16, height = 3)
ggsave("./prediction_traj/sad_traj.pdf", pred.traj.plot(sad.all.res[[3]]$res) + ggtitle("Sadness"), width = 16, height = 3/14*18)
ggsave("./prediction_traj/nerv_traj.pdf",pred.traj.plot(nerv.all.res[[3]]$res) + ggtitle("Nervousness"), width = 16, height = 3/14*18)

roc1 = roc(ang.all.res[[3]]$res$y, ang.all.res[[3]]$res$p.stack)
roc2 = roc(sad.all.res[[3]]$res$y, sad.all.res[[3]]$res$p.stack)
roc3 = roc(nerv.all.res[[3]]$res$y, nerv.all.res[[3]]$res$p.stack)

idx.1 = order((1-roc1$sensitivities)^2 + (1-roc1$specificities)^2)[1]
idx.2 = order((1-roc2$sensitivities)^2 + (1-roc2$specificities)^2)[1]
idx.3 = order((1-roc3$sensitivities)^2 + (1-roc3$specificities)^2)[1]

c( roc1$sensitivities[idx.1], roc1$specificities[idx.1] )
c( roc2$sensitivities[idx.2], roc2$specificities[idx.2] )
c( roc3$sensitivities[idx.3], roc3$specificities[idx.3] )
#### feature importance ####
var.map = list("Power" = "Phone (MinsHr)",
               "ActScore" = "Accelerometer Score",
               "SleepOnset_time" = "Sleep Onset Time",
               "mphuse" = "Phone (MinsDay)",
               "hphuse" = "Phone (Hr)",
               "Wake_time" = "Wake Time",
               "missing" = "GPS Available",
               "percentHome" = "Daily Home %",
               "Sleep_Duration" = "Sleep Duration",
               "radiusMobility" = "Daily Mobility Area",
               "numPlaces" = "Daily Visited #",
               "TimeDay" = "Time of Day",
               "homDist" = "Distance from Home",
               "GPS_freq" = "Hourly Visited #")

partial.bin = function(model, data, var, n.points = 50){
  all.values = seq(min(data[[var]]), max(data[[var]]), length.out = 50)
  all.p = sapply( all.values, function(ind.value){
    data.tmp = data
    data.tmp[[var]] = ind.value
    all.pred = predict(model, data.tmp, type = "prob")[,2]
    mean(all.pred)
  } )
  cbind(all.values, all.p)
}

plot.partial.bin = function(all.import, data.split, name){
  var.extreme = paste(name, "extreme", sep = ".")
  colnames(all.import) = names(data.split)
  var.order = order(rowMeans(abs(all.import), na.rm = T), decreasing = F)
  sub.order = names(sort(sapply(data.split, function(x) mean(x[[var.extreme]]==1)), decreasing = T))
  plt.import = melt(all.import)
  plt.import$Var1 = factor(plt.import$Var1, levels = row.names(all.import)[var.order], ordered = T)
  plt.import$Var2 = factor(plt.import$Var2, levels = sub.order, ordered = T)
  p1 = ggplot(aes(x = Var2, y = Var1, fill = value), data = plt.import) + geom_tile(color = "black") + theme_bw() +
    scale_fill_gradient2(limits = c(-0.25,0.25), low = "blue", high = "red", mid = "white", name = "Importance") + 
    theme(axis.text.x = element_text(angle = 45, hjust=1)) + 
    xlab("Subject") + ylab("Variable") + 
    scale_x_discrete(expand = c(0,0)) + scale_y_discrete(expand = c(0,0)) + coord_fixed()
  return(p1)
}

plot.importance = function(data.use, name, n.filter = 4){
  outcome = paste(name, "extreme", sep = ".")
  data.filter = data.use[sapply(data.use, function(x) sum(x[[outcome]])>n.filter)]
  data.merge = do.call(rbind, data.filter)
  import.res = get.import(data.merge, name)
  row.names(import.res) = sapply(row.names(import.res), function(x) var.map[[x]])
  plot.partial.bin(import.res, data.filter, name)
}

mean(apply(import.res, 2, function(x) rank(-abs(x)))[1,]<=3)

ang.import = plot.importance(data.ang.use, "ang") + ggtitle("Anger")
sad.import = plot.importance(data.sad.use, "sad") + ggtitle("Sadness")
nerv.import = plot.importance(data.nerv.use, "nev") + ggtitle("Nervousness")
ggsave("./feature_importance/ang_import.pdf", ang.import, height = 4, width = 6)
ggsave("./feature_importance/sad_import.pdf", sad.import, height = 4, width = 6*19/14)
ggsave("./feature_importance/nerv_import.pdf", nerv.import, height = 4, width = 6*19/14)

#### time-series CV ####
ang.all.res.tscv = analysis.wrapper("ang.extreme", data.ang.use, cv = "ts")
sad.all.res.tscv = analysis.wrapper("sad.extreme", data.sad.use, cv = "ts")
nerv.all.res.tscv = analysis.wrapper("nev.extreme", data.nerv.use, cv = "ts")

all.methods = c("PEM-ENet", "PEM-SVM", "PEM-RF", "PDEM")
plot.roc.tscv = function(all.res){
  plt.roc = do.call(rbind, lapply(1:4, function(idx){
    if(!is.na(all.res[[idx]])[1]){
      roc.res = roc(all.res[[idx]]$y, all.res[[idx]]$p.stack, direction = "<")
      data.frame(sensitivities = roc.res$sensitivities, 
                 specificities = roc.res$specificities, 
                 Method = all.methods[idx])
    }
  }))
  ggplot(plt.roc, aes(x = 1-specificities, y = sensitivities, color = Method)) + geom_path() + theme_bw() + 
    geom_abline(slope = 1, intercept = 0, linetype = 2) + xlab("1 - Specificity") + ylab("Sensitivity") +
    scale_color_manual(values = c("#93AA00", "#00BA38", "#00B9E3", "#DB72FB")) + coord_fixed()
}

ggsave("./AUC_res/ang_auc_tscv.pdf", plot.roc.tscv(ang.all.res.tscv) + ggtitle("Anger"), width = 6, height = 5)
ggsave("./AUC_res/sad_auc_tscv.pdf", plot.roc.tscv(sad.all.res.tscv) + ggtitle("Sadness"), width = 6, height = 5)
ggsave("./AUC_res/nerv_auc_tscv.pdf", plot.roc.tscv(nerv.all.res.tscv) + ggtitle("Nervousness"), width = 6, height = 5)

cbind( sapply(ang.all.res.tscv[-5], function(x){
  if(!is.na(x)[1]){
    auc(x$y, x$p.stack, direction = "<")
  }
}), sapply(sad.all.res.tscv[-5], function(x){
  if(!is.na(x)[1]){
    auc(x$y, x$p.stack, direction = "<")
  }
}), sapply(nerv.all.res.tscv[-5], function(x){
  if(!is.na(x)[1]){
    auc(x$y, x$p.stack, direction = "<")
  }
}))


#### proportion of positive predictions ####
pprop.mt = sapply(seq(0,0.99,0.01), function(x){
  c(mean(ang.all.res[[3]]$res$p.stack>x),
    mean(sad.all.res[[3]]$res$p.stack>x),
    mean(nerv.all.res[[3]]$res$p.stack>x))
})
plt.pprop = data.frame(Threshold = rep(seq(0,0.99,0.01), 3),
                       pprop = c(t(pprop.mt)),
                       Emotion = factor(rep(c("Anger", "Sadness", "Nervousness"), each = 100), 
                                        levels = c("Anger", "Sadness", "Nervousness"),
                                        ordered = T))
ggsave("./predicted_positive_proportion.pdf", 
       ggplot(plt.pprop, aes(x = Threshold, y = pprop, color = Emotion)) + geom_line() +
  scale_color_discrete() + ylab("Proportion of HNA") +theme_bw(),
  width = 6, height = 4)
