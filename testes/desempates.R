# Como o R desempata: casos feitos à mão, com empates forçados. Chamado por t_desempates.py.
# Uso: Rscript desempates.R <saida.txt>
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({ library(MASS); library(class); library(rpart); library(e1071); library(randomForest); library(kknn) })
sai <- file(args[1], "w")
say <- function(...) writeLines(paste0(...), sai)
ctl <- function(L = 1, prof = 30) rpart.control(minsplit = 2 * L, minbucket = L, cp = -1, maxdepth = prof, maxcompete = 0, maxsurrogate = 0, xval = 0)
raiz <- function(m) { f <- m$frame; if (nrow(f) < 2) return("sem divisão"); s <- m$splits; paste0(rownames(s)[1], " ", ifelse(s[1, "ncat"] < 0, "<", ">="), " ", s[1, "index"], " vai para a esquerda") }

say("## rpart")
# 1. empate entre atributos: a mesma partição por X1 ou por X2
d <- data.frame(x1 = c(0, .1, 1, 1.1), x2 = c(0, .1, 1, 1.1), y = factor(c(1, 1, 2, 2)))
say("atributos empatados, y ~ x1 + x2: ", raiz(rpart(y ~ x1 + x2, d, method = "class", control = ctl())))
say("atributos empatados, y ~ x2 + x1: ", raiz(rpart(y ~ x2 + x1, d, method = "class", control = ctl())))
d2 <- d[c(3, 1, 4, 2), ]
say("atributos empatados, linhas embaralhadas: ", raiz(rpart(y ~ x1 + x2, d2, method = "class", control = ctl())))
# 2. empate entre limiares do mesmo atributo: 0,5 e 2,5 dão o mesmo ganho
d <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(5, 5, 5, 5), y = factor(c(1, 2, 2, 1)))
say("limiares empatados (0,5 ou 2,5): ", raiz(rpart(y ~ x1 + x2, d, method = "class", control = ctl(1, 1))))
d <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(5, 5, 5, 5), y = factor(c(2, 1, 1, 2)))
say("limiares empatados, rótulos trocados: ", raiz(rpart(y ~ x1 + x2, d, method = "class", control = ctl(1, 1))))
d <- data.frame(x1 = c(0, 1, 2, 3, 4, 5), x2 = 5, y = factor(c(1, 1, 2, 2, 1, 1)))
say("limiares empatados 1,5 e 3,5: ", raiz(rpart(y ~ x1 + x2, d, method = "class", control = ctl(1, 1))))
# 3. folha com contagens empatadas
d <- data.frame(x1 = c(0, 0), x2 = c(0, 0), y = factor(c(1, 2)))
m <- rpart(y ~ x1 + x2, d, method = "class", control = ctl())
say("folha 1×1, níveis 1,2: prevê ", as.character(predict(m, d[1, ], type = "class")))
d$y <- factor(c(1, 2), levels = c(2, 1))
m <- rpart(y ~ x1 + x2, d, method = "class", control = ctl())
say("folha 1×1, níveis 2,1: prevê ", as.character(predict(m, d[1, ], type = "class")))
d <- data.frame(x1 = c(0, 0, 0), x2 = c(0, 0, 0), y = factor(c(3, 2, 1)))
m <- rpart(y ~ x1 + x2, d, method = "class", control = ctl())
say("folha 1×1×1, três classes: prevê ", as.character(predict(m, d[1, ], type = "class")))
# 4. divisão com ganho zero (XOR com quatro pontos)
d <- data.frame(x1 = c(0, 1, 0, 1), x2 = c(0, 1, 1, 0), y = factor(c(1, 1, 2, 2)))
m <- rpart(y ~ x1 + x2, d, method = "class", control = ctl())
say("XOR de 4 pontos (ganho zero na raiz): ", nrow(m$frame), " nós; raiz: ", raiz(m))
m <- rpart(y ~ x1 + x2, d, method = "class", parms = list(split = "information"), control = ctl())
say("XOR de 4 pontos, entropia: ", nrow(m$frame), " nós")
# 5. a condição da divisão: "<" ou ">=" à esquerda
d <- data.frame(x1 = c(0, 1, 2, 3), x2 = 5, y = factor(c(2, 2, 1, 1)))
say("direção, classe 2 à esquerda no eixo: ", raiz(rpart(y ~ x1 + x2, d, method = "class", control = ctl(1, 1))))
d$y <- factor(c(1, 1, 2, 2))
say("direção, classe 1 à esquerda no eixo: ", raiz(rpart(y ~ x1 + x2, d, method = "class", control = ctl(1, 1))))

say("## randomForest")
# empate de votos: o que diz a documentação de predict.randomForest
h <- tools::Rd2txt(utils:::.getHelpFile(help("predict.randomForest", package = "randomForest")), out = tempfile())
say("documentação de predict.randomForest: ", paste(trimws(grep("ties", readLines(h), value = TRUE, ignore.case = TRUE)), collapse = " "))
# nodesize: menor nó que chegou a ser dividido e maior folha impura, sem bootstrap (todos os pontos em cada árvore)
set.seed(3)
n <- 80
d <- data.frame(x1 = runif(n), x2 = runif(n))
d$y <- factor(ifelse(d$x1 + .4 * rnorm(n) > .5, 1, 2))
for (L in c(1, 2, 3, 5, 8, 12)) {
  m <- randomForest(y ~ x1 + x2, d, ntree = 30, mtry = 2, nodesize = L, replace = FALSE, sampsize = n)
  minInt <- Inf; maxImpura <- 0; minFolha <- Inf
  for (t in 1:30) {
    g <- getTree(m, t)
    rota <- function(i, idx) {
      if (g[i, "status"] == -1) { minFolha <<- min(minFolha, length(idx)); if (length(unique(d$y[idx])) > 1) maxImpura <<- max(maxImpura, length(idx)); return(invisible()) }
      minInt <<- min(minInt, length(idx))
      v <- d[idx, g[i, "split var"]]
      rota(g[i, "left daughter"], idx[v <= g[i, "split point"]]); rota(g[i, "right daughter"], idx[v > g[i, "split point"]])
    }
    rota(1, 1:n)
  }
  say("nodesize = ", L, ": menor nó dividido = ", minInt, "; maior folha impura = ", maxImpura, "; menor folha = ", minFolha)
}

say("## class::knn")
tr <- data.frame(x1 = c(1, -1, 0), x2 = c(0, 0, 1)); cl <- factor(c(1, 2, 2))
res <- replicate(200, as.character(knn(tr[1:2, ], data.frame(x1 = 0, x2 = 0), cl[1:2], k = 2)))
say("voto 1×1 (k = 2, 200 vezes): ", paste(names(table(res)), table(res), sep = ":", collapse = " "))
res <- replicate(200, as.character(knn(tr, data.frame(x1 = 0, x2 = 0), cl, k = 1, use.all = TRUE)))
say("três à mesma distância, k = 1, use.all = TRUE: ", paste(names(table(res)), table(res), sep = ":", collapse = " "))

say("## kknn")
kk <- function(tr, te, k, ker, p = 2) { m <- kknn(y ~ ., tr, te, k = k, distance = p, kernel = ker, scale = FALSE); list(cls = as.character(fitted(m)), prob = m$prob, C = m$C, D = m$D, W = m$W) }
te <- data.frame(x1 = 0, x2 = 0)
# voto empatado 1×1 com k = 2 < n (um terceiro ponto longe); muda a ordem das linhas e a dos níveis
for (lv in list(c(1, 2), c(2, 1))) for (o in list(1:3, c(2, 1, 3))) for (ker in c("rectangular", "inv")) {
  tr <- data.frame(x1 = c(1, -1, 5), x2 = c(0, 0, 0), y = factor(c(1, 2, 1), levels = lv))[o, ]
  r <- kk(tr, te, 2, ker)
  say("voto empatado 1×1, níveis ", paste(lv, collapse = ","), ", primeira linha da classe ", as.character(tr$y[1]), ", kernel ", ker, ": classe ", r$cls)
}
# vizinho mais próximo decide o empate no "rectangular": classe 2 mais perto
tr <- data.frame(x1 = c(1, -.5, 2, -2, 9), x2 = 0, y = factor(c(1, 2, 1, 2, 1)))
say("voto 2×2 (k = 4), classe 2 com o vizinho mais próximo, rectangular: classe ", kk(tr, te, 4, "rectangular")$cls)
tr <- data.frame(x1 = c(1, -1, 0, 3), x2 = c(0, 0, 1, 3), y = factor(c(1, 2, 2, 1)))
for (o in list(1:4, c(2, 1, 3, 4), c(3, 2, 1, 4))) { r <- kk(tr[o, ], te, 1, "rectangular"); say("três à mesma distância, k = 1, ordem ", paste(o, collapse = ","), ": vizinho = linha ", o[r$C[1]], ", classe ", r$cls) }
for (p in c(1, 2, 3)) { r <- kk(tr, te, 2, "rectangular", p); say("distâncias com p = ", p, ": ", paste(signif(r$D, 6), collapse = " ")) }
tr <- data.frame(x1 = c(0, 1, -2), x2 = c(0, 0, 0), y = factor(c(2, 1, 1)))
r <- kk(tr, te, 2, "inv")
say("distância zero, kernel inv, k = 2: classe ", r$cls, "; pesos ", paste(signif(r$W, 6), collapse = " "), "; prob ", paste(signif(r$prob, 6), collapse = " "))
tr <- data.frame(x1 = c(1, -2, 0), x2 = c(0, 0, 2), y = factor(c(1, 2, 2)))
r <- kk(tr, te, 2, "inv")
say("pesos 1/d vs kernel inv (d = 1, 2; d(3) = 2): pesos ", paste(signif(r$W, 8), collapse = " "), "; prob ", paste(signif(r$prob, 8), collapse = " "))
tr <- data.frame(x1 = c(1, -2, 0, 5), x2 = c(0, 0, 3, 0), y = factor(c(1, 2, 2, 1)))
r <- kk(tr, te, 2, "inv")
say("pesos 1/d vs kernel inv (d = 1, 2; d(3) = 3): pesos ", paste(signif(r$W, 8), collapse = " "), "; prob ", paste(signif(r$prob, 8), collapse = " "))
# k = n: não existe o (k+1)-ésimo vizinho; com "inv" os pesos deveriam ser 1/d = 1, 1/2, 1/3
tr <- data.frame(x1 = c(1, -2, 3), x2 = c(0, 0, 0), y = factor(c(1, 2, 2)))
r <- kk(tr, te, 3, "inv")
say("k = n = 3, kernel inv (d = 1, 2, 3; 1/d dá prob. 0,545 0,455): pesos ", paste(signif(r$W, 6), collapse = " "), "; prob ", paste(signif(r$prob, 6), collapse = " "))
tr <- data.frame(x1 = c(1, -1, 0, 3), x2 = c(7, 7, 7, 7), y = factor(c(1, 2, 2, 1)))
r <- kk(tr, data.frame(x1 = 0, x2 = 0), 1, "rectangular")
say("atributo constante no treino (x2 = 7), teste em x2 = 0: D = ", paste(signif(r$D, 6), collapse = " "))

say("## MASS, e1071")
say("predict.lda usa max.col: ", any(grepl("max.col", deparse(getAnywhere("predict.lda")$objs[[1]]))))
say("predict.qda usa max.col: ", any(grepl("max.col", deparse(getAnywhere("predict.qda")$objs[[1]]))))
say("predict.naiveBayes usa which.max: ", any(grepl("which.max", deparse(getAnywhere("predict.naiveBayes")$objs[[1]]))))
d <- data.frame(x1 = c(-1, -1.2, 1, 1.2), x2 = c(0, .3, 0, .3), y = factor(c(1, 1, 2, 2)))
m <- lda(y ~ x1 + x2, d); g <- data.frame(x1 = 0, x2 = .15)
say("lda, ponto equidistante: posterior ", paste(signif(predict(m, g)$posterior, 8), collapse = " "), "; 200 previsões: ", paste(names(table(replicate(200, as.character(predict(m, g)$class)))), table(replicate(200, as.character(predict(m, g)$class))), sep = ":", collapse = " "))
m <- naiveBayes(y ~ x1 + x2, d)
say("naiveBayes, ponto equidistante: ", paste(signif(predict(m, g, type = "raw"), 8), collapse = " "), " classe ", as.character(predict(m, g)))

say("## e1071::svm, f(x) = 0")
d <- data.frame(x1 = c(-1, 1), x2 = c(0, 0), y = factor(c(-1, 1)))
for (o in list(1:2, 2:1)) {
  m <- svm(y ~ x1 + x2, d[o, ], kernel = "linear", cost = 1, scale = FALSE)
  pr <- predict(m, data.frame(x1 = 0, x2 = 0), decision.values = TRUE)
  say("ordem ", paste(o, collapse = ","), ": rho = ", m$rho, ", f(0) = ", attr(pr, "decision.values")[1], " -> classe ", as.character(pr))
}
close(sai)
