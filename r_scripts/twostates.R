rm(list=ls())

library(rcraam)
library(ggplot2)
library(dplyr)
library(latex2exp)
loadNamespace("reshape2")
loadNamespace("tidyr")
theme_set(theme_light())


source("libs.R")
# not sure why this line doesn't work for me. It doesn't source the file into my
# workspace
# debugSource("libs.R")

twostates <-
"idstatefrom,idaction,idstateto,probability,reward
0,0,0,0.8,0
0,0,1,0.2,0
1,0,0,0.5,0
1,0,1,0.5,0
0,1,0,0.1,1
0,1,1,0.9,1
1,1,0,0.5,0
1,1,1,0.5,0"

twostates.mdp <- read.csv(text=twostates)
nstates <- (twostates.mdp %>% select(idstatefrom) %>% unique() %>% count())[[1]]

Pinit <- cbind(c(1,0)) # it has a wierd dimension
discount <- 0.9


# solve for the optimal policy
optimal <- solve_mdp(twostates.mdp, discount)
Popt <- matrix_mdp_transition(twostates.mdp, optimal$policy)$P
eps_scalar <- 0.3

epsilon <- eps_scalar * rep(1, ncol(Pinit))

# inverse rl part
features.csv <- 
"idstate,idaction,f1,f2,f3,f4
0,0,1,0,0,0
0,1,0,1,0,0
1,0,0,0,1,0
1,1,0,0,0,1"
nfeatures <- 4

features.frame <- read.csv(text=features.csv)

Ab <- matrix_mdp_lp(twostates.mdp, discount)
A <- Ab$A
b <- Ab$b
idstateaction <- Ab$idstateaction

# construct the occupancy frequencies
  # matrix that translates state occ freqs to state-action occ freqs
Pi <- 
  full_join(idstateaction, 
          optimal$policy %>% mutate(val = 1), 
          by = c("idstate", "idaction")) %>%
    tidyr::replace_na(list(val = 0)) %>%
    select(row_index, idstate, val) %>%
    reshape2::acast(row_index ~ idstate, value.var = "val") %>%
    tidyr::replace_na(0)

U <- Pi %*% solve(t(diag(c(1,1)) - discount * Popt), Pinit)

# construct the feature matrix
Phi <-
  full_join(features.frame, idstateaction, by = c("idstate","idaction")) %>%
    arrange(row_index) %>% select(-idstate, -idaction, -row_index) %>%
    as.matrix()

# construct constraints on the rewards that are consistent with the demonstration
# variables: w, l, v
# constraints:
#         w -      l               <= 0
#       - w -      l               <= 0
#              1^T l               <= 1
# - U' Phi w          +  Pinit' v  <= epsilon
#     Phi w           -       A v  <= 0

# Construct the constraint matrix
ninitial <- ncol(Pinit)

Iw <- diag(1,nfeatures)
Zw <- matrix(0, nfeatures, nfeatures)
Z1w <- matrix(0, 1, nfeatures)
v1l <- matrix(1, 1, nfeatures)
Il <- Iw
Zl <- matrix(0,ninitial, nfeatures)
Zal <- matrix(0,nrow(A),nfeatures)
Iv <- diag(1,nstates)
Zv <- matrix(0, nstates, nstates)
Z1v <- matrix(0, 1, nstates)

const.matrix <- 
  qm(          Iw,  -Il,       Zv |
              -Iw,  -Il,       Zv |
              Z1w,  v1l,      Z1v |
   - t(U) %*% Phi,   Zl, t(Pinit) |
              Phi,  Zal,      -A  )

const.lower <- 
  c(rep(0,nfeatures*2),1,epsilon,rep(0, nrow(A)))


# projection matrix for the first two dimensions
proj_w <- diag(1,ncol(const.matrix),2)
 
points <- extreme.points.ineq(const.matrix, const.lower, proj_w)
points <- rbind(points, points[1,])

l1ball_points <- qm(-1, 0 | 0, 1 | 1 , 0 | 0, -1 | -1, 0) %>% as.data.frame()
g <- ggplot() + 
  geom_polygon(aes(x=p1, y=p2), fill="green", alpha=0.1, data=points) +
  geom_path(aes(x=p1, y=p2), color="dark green", linetype="dashed",  data=points) +
  geom_path(aes(x=V1,y=V2), color="black", data = l1ball_points) +
  labs(x = TeX("Weight w_1"), y = TeX("Weight w_2"))

print(g)

# Questions:
# - How does this set compare to the tightest possible ambiguity set coming from
#   the Bellman equation?

## The Bellman way

# construct T matrix
Abo <- matrix_mdp_lp(twostates.mdp, 1)
Abz <- matrix_mdp_lp(twostates.mdp, 0)
T <- -Abo$A + Abz$A

# construct E matrix
Exp <- optimal$policy %>% rename(idactionexp = idaction) # this is the experience

# filter states from all state-action pairs based on the experience
# to get the rows of E
Erows <- 
  inner_join(Exp, idstateaction, by="idstate") %>%
    mutate(row_index = row_number())
Ecols <- 
  idstateaction %>% rename(idstate_col=idstate, idaction_col=idaction, col_index=row_index)

E.df <- tidyr::crossing(Erows,Ecols) %>%
  mutate(value = as.numeric((idstate_col == idstate) & (idaction_col == idactionexp)) - 
           as.numeric((idstate_col == idstate) & (idaction_col == idaction)) )

E <- reshape2::acast(E.df, row_index ~ col_index, value.var = "value")

# remove all rows of E that sum to 0

E <- E[rowSums(abs(E)) > 0,, drop = FALSE]

# construct constraints on the rewards that are consistent with the demonstration
# variables: w, l, v
# constraints:
#         w -      l                <= 0
#       - w -      l                <= 0
#              1^T l                <= 1
# - E Phi w          -  gamma E T v <= epsilon
#     Phi w          -          A v <= 0

Zel <- matrix(0, nrow(E), nfeatures)
Oe <- rep(1, nrow(E))

const.matrix <- 
  qm(          Iw,  -Il,                  Zv |
              -Iw,  -Il,                  Zv |
              Z1w,  v1l,                 Z1v |
      - E %*% Phi,  Zel, -discount * E %*% T |
              Phi,  Zal,                  -A )

const.lower <- 
  c(rep(0,nfeatures*2),1,eps_scalar * Oe,rep(0, nrow(A)))


# projection matrix for the first two dimensions
proj_w <- diag(1,ncol(const.matrix),2)

points_bell <- extreme.points.ineq(const.matrix, const.lower, proj_w)
points_bell <- rbind(points_bell, points_bell[1,])

g <- ggplot() + 
  geom_polygon(aes(x=p1, y=p2), fill="red", alpha=0.1, data=points_bell) +
  geom_path(aes(x=p1, y=p2), color="dark red", linetype="dashed",  data=points_bell) +
  geom_path(aes(x=V1,y=V2), color="black", data = l1ball_points) +
  labs(x = TeX("Weight w_1"), y = TeX("Weight w_2"))

plot(g)

