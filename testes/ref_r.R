# Referências em R para os testes (MASS, e1071, class, rpart, randomForest).
# Uso: Rscript ref_r.R <metodo> <entrada.csv> <grade.csv> <saida.csv>
# entrada: caso,x1,x2,y ; grade: x1,x2 (a mesma para todos os casos); saída: caso,erro,<probabilidades da grade em colunas p1..pK e classe prev>
args <- commandArgs(trailingOnly = TRUE)
metodo <- args[1]; ent <- read.csv(args[2]); grade <- read.csv(args[3]); saida <- args[4]
extra <- if (length(args) >= 5) read.csv(args[5]) else NULL
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(MASS); library(class); library(rpart)
  if (metodo == "nb") library(e1071)
  if (metodo == "rf") library(randomForest)
  if (metodo == "kknn") suppressWarnings(library(kknn))
})
out <- file(saida, "w")
writeLines("caso,linha,erro,pred", out)
for (cs in unique(ent$caso)) {
  d <- ent[ent$caso == cs, c("x1", "x2", "y")]
  d$y <- factor(d$y)
  g <- grade
  res <- tryCatch({
    if (metodo == "lda") {
      m <- suppressWarnings(lda(y ~ x1 + x2, data = d))
      as.integer(as.character(predict(m, g)$class))
    } else if (metodo == "qda") {
      m <- suppressWarnings(qda(y ~ x1 + x2, data = d))
      as.integer(as.character(predict(m, g)$class))
    } else if (metodo == "nb") {
      m <- naiveBayes(y ~ x1 + x2, data = d)
      as.integer(as.character(predict(m, g, type = "class")))
    } else if (metodo == "knn") {
      k <- extra$K[extra$caso == cs]
      set.seed(1)
      as.integer(as.character(knn(train = d[, c("x1", "x2")], test = g, cl = d$y, k = k, use.all = TRUE)))
    } else if (metodo == "kknn") {
      e <- extra[extra$caso == cs, ]
      m <- kknn(y ~ ., train = d, test = g, k = e$K, distance = e$p, kernel = as.character(e$kernel), scale = FALSE)
      as.integer(as.character(fitted(m)))
    } else if (metodo == "rpart") {
      L <- extra$L[extra$caso == cs]; prof <- extra$prof[extra$caso == cs]; crit <- as.character(extra$crit[extra$caso == cs])
      m <- rpart(y ~ x1 + x2, data = d, method = "class", parms = list(split = crit), control = rpart.control(minsplit = 2 * L, minbucket = L, cp = -1, maxdepth = prof, maxcompete = 0, maxsurrogate = 0, xval = 0))
      as.integer(as.character(predict(m, g, type = "class")))
    } else if (metodo == "rf") {
      ns <- if ("nodesize" %in% names(extra)) extra$nodesize[extra$caso == cs] else 1
      m <- randomForest(y ~ x1 + x2, data = d, ntree = extra$B[extra$caso == cs], mtry = extra$m[extra$caso == cs], nodesize = ns, replace = TRUE)
      as.integer(as.character(predict(m, g, type = "response")))
    }
  }, error = function(e) e)
  if (inherits(res, "error")) {
    msg <- gsub('[",\n]', " ", conditionMessage(res))
    writeLines(sprintf("%s,0,%s,", cs, msg), out)
  } else {
    writeLines(sprintf("%s,%d,,%d", cs, seq_along(res), res), out)
  }
}
close(out)
