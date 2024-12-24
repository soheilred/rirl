library("dplyr")
library("combinat")
library("functional")


#' Constructs the extreme points of a polytope
#' defined by A x <= b
#' 
#' @param A coefficients
#' @param b right-hand side
#' @param simplex.cstr whether to add simplex constraints
#' @param P Projects from ncol(A) dimensions to 2
#' @seealso simplex.projections
extreme.points.ineq <- function(A, b, P){
  requireNamespace("combinat")
  requireNamespace("functional")
  
  Ap <- A
  bp <- b

  # iterate over all possible subsets of inequalities to determine the extreme points
  combos <- combinat::combn(nrow(Ap),ncol(Ap))  # vary the number of basic rows based on # vars
  subselect <- function(i){
    list(Ap[combos[,i],], bp[combos[,i]])
  }
  # compute basic solutions that are also feasible
  solveit <- function(Ab){
    A <- Ab[[1]]; b <- Ab[[2]]
    if(Matrix::rankMatrix(A) == ncol(A)){
      solve(A, b)    
    }
    else
      NULL
  }
  # check feasibility
  isfeasible <- function(p){
    # the -0.. is to prevent numerical issues
    if(!is.null(p) && all(Ap %*% p - 0.00001 <= bp)){   p       }
    else{   NULL  }
  }
  
  points <- Filter(Negate(is.null),  lapply(1:ncol(combos), 
                  functional::Compose(subselect, solveit, isfeasible))) 
  
  if(length(points) == 0){
    return(NULL)
  }
  points <- points %>% simplify2array() %>% t
  points <- points %*% P
  p.hull <- chull(points)
  if(!is.null(P)){
    return(points[p.hull,,drop=FALSE] %>% as.data.frame() %>% transmute(p1=V1, p2=V2))
  }else{
    return(points[p.hull,,drop=FALSE] %>% as.data.frame() %>% transmute(p1=V1, p2=V2, p3=V3))
  }
  
}

# from http://www.sumsar.net/blog/2014/03/a-hack-to-create-matrices-in-R-matlab-style/
# does string parsing: make sure to not use | for anything else but a row separator
qm<-function(...)
{
  # turn ... into string
  args<-deparse(substitute(rbind(cbind(...))))
  
  # create "rbind(cbind(.),cbind(.),.)" construct
  args<-gsub("\\|","),cbind(",args)
  
  # eval
  # marek: added envir=parent.frame(1) to access local function args
  eval(parse(text=args),envir=parent.frame())
}
