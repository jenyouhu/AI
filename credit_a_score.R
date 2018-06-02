#载入需要的R包
library('grid')
library('DMwR')#knnImputation
library('corrplot')#corrplot
library('iterators')#迭代，caret依赖包
library('caret')#createDataPartition（数据分割功能）
library('pROC')#modelroc（用于分类器比较）
library('VIM') #matrixplot

library(InformationValue) # IV / WOE calculation
library(ggplot2)
library(GGally)
library(scales)
library(lattice)
library(MASS)
library(memisc)
library(Rcpp)#使用c++语言
library(Amelia)#有缺失值绘图的函数
library(gridExtra)
library(tidyr)
library(mice)
library(dplyr)

library(splines) #数据差值包
library('randomForest') # 分类算法
library("rpart")
library('ggthemes') # 可视化
library(stringr)




#读取数据集
cs_training<-read.csv('cs_training.csv')

#了解数据维度
dim(cs_training)

#数据字段
names(cs_training)

#数据结构
str(cs_training)

# 数据的列名、行名和数据结构
attributes(cs_training) 

#去掉第一行 
cs_training<-cs_training[,-1]
View(cs_training)
head(cs_training)

#把变量映射成 y 与 x (重命名)
names(cs_training)<-c("y",paste("x",1:10,sep=""))
str(cs_training)
View(cs_training)

#查看数据集缺失数据情况

matrixplot(cs_training)
md.pattern(cs_training)

missmap(cs_training,main='Missing values')

#x5(MonthlyIncome)缺失值处理(使用中位数)
cs_training$x5 <- na.roughfix(cs_training$x5)

#使用KNN方法对缺失值进行填补
#traindata<-knnImputation(cs_training,k=10,meth = "weighAvg")


#x10(NumberOfDependents)3924个缺失值，所占比重3924/150000不大，
#故直接删除
cs_training <- cs_training[!is.na(cs_training$x10),]

#对x2变量(客户的年龄)定量分析
unique(cs_training$x2)

#删除年龄为0的数据 
cs_training<-cs_training[-which(cs_training$x2==0),]
unique(cs_training$x2)

#画出箱线图,找出异常值
boxplot(cs_training$x3,cs_training$x7,cs_training$x9)

#对x3，x7,x9 变量(客户的年龄)定量分析
unique(cs_training$x3)
unique(cs_training$x7)
unique(cs_training$x9)

#去掉异常值96和98
#因为有96和98值的x3、x7、x9是在同一行，所以动一个变量即可
cs_training<-cs_training[-which(cs_training$x3==96),]
cs_training<-cs_training[-which(cs_training$x3==98),]

#单变量分析，直方图显示，年龄\收入呈正太分布
hist(cs_training$x2,freq=F)
hist(cs_training$x5/100,freq=F)

ggplot(cs_training, aes(x = x2, y = ..density..)) +
  geom_histogram(fill = "blue", colour = "grey60",
                 size = 0.2, alpha = 0.2) + geom_density()

ggplot(cs_training, aes(x = x5, y = ..density..)) +
  geom_histogram(fill = "blue", colour = "grey60",
                 size = 0.2, alpha = 0.2) + geom_density() +
  xlim(1, 20000)

#变量相关性分析 
cor1<-cor(cs_training[,1:11])
corrplot(cor1)
corrplot(cor1,method = "number")


#切分数据表
table(cs_training$y)

#由上表看出，对于响应变量SeriousDlqin2yrs，
#存在明显的类失衡问题，SeriousDlqin2yrs等于1的观测为9712，
#仅为所有观测值的6.6%。因此我们需要对非平衡数据进行处理，
#在这里可以采用SMOTE算法，用R对稀有事件进行超级采样。

#我们利用caret包中的createDataPartition
#（数据分割功能）函数将数据随机分成相同的两份。

set.seed(1234) 

splitIndex<-createDataPartition(cs_training$y,time=1,p=0.5,list=FALSE)
train<-cs_training[splitIndex,]
test<-cs_training[-splitIndex,]

#平衡分类结果如下：
prop.table(table(train$y))
prop.table(table(test$y))
table(train$y)
table(test$y)


#首先利用glm函数对所有变量进行Logistic回归建模，模型如下
fit<-glm(y~.,train,family = "binomial")
summary(fit)

#可以看出，利用全变量进行回归，模型拟合效果并不是很好，
#其中x1,x4,x6三个变量的p值未能通过检验，在此直接剔除这三个变量，
#利用剩余的变量对y进行回归。

fit2<-glm(y~x2+x3+x5+x7+x8+x9+x10,train,family = "binomial")
summary(fit2)

#第二个回归模型所有变量都通过了检验(fit2)
#甚至AIC值（赤池信息准则）更小，所有模型的拟合效果更好些。

#模型评估
#通常一个二值分类器可以通过ROC（Receiver Operating Characteristic）曲线和AUC值来评价优劣。
#很多二元分类器会产生一个概率预测值，而非仅仅是0-1预测值。我们可以使用某个临界点（例如0.5），以划分哪些预测为1，哪些预测为0。得到二元预测值后，可以构建一个混淆矩阵来评价二元分类器的预测效果。所有的训练数据都会落入这个矩阵中，而对角线上的数字代表了预测正确的数目，即true positive + true nagetive。
#同时可以相应算出TPR（真正率或称为灵敏度）和TNR（真负率或称为特异度）。我们主观上希望这两个指标越大越好，但可惜二者是一个此消彼涨的关系。除了分类器的训练参数，临界点的选择，也会大大的影响TPR和TNR。有时可以根据具体问题和需要，来选择具体的临界点。
#如果我们选择一系列的临界点，就会得到一系列的TPR和TNR，将这些值对应的点连接起来，就构成了ROC曲线。ROC曲线可以帮助我们清楚的了解到这个分类器的性能表现，还能方便比较不同分类器的性能。在绘制ROC曲线的时候，习惯上是使用1-TNR作为横坐标即FPR（false positive rate），TPR作为纵坐标。这是就形成了ROC曲线。
#而AUC（Area Under Curve）被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

pre <- predict(fit2,test)

#在R中，可以利用pROC包，它能方便比较两个分类器，
#还能自动标注出最优的临界点，图看起来也比较漂亮。
#在下图中最优点FPR=1-TNR=0.845，TPR=0.638，AUC值为0.8102，说明该模型的预测效果还是不错的，正确较高。
modelroc <- roc(test$y,pre)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

#六、WOE转换
#证据权重（Weight of Evidence,WOE）转换可以将Logistic回归模型转变为标准评分卡格式。引入WOE转换的目的并不是为了提高模型质量，只是一些变量不应该被纳入模型，这或者是因为它们不能增加模型值，或者是因为与其模型相关系数有关的误差较大，其实建立标准信用评分卡也可以不采用WOE转换。这种情况下，Logistic回归模型需要处理更大数量的自变量。尽管这样会增加建模程序的复杂性，但最终得到的评分卡都是一样的。
#用WOE(x)替换变量x。WOE()=ln[(违约/总违约)/(正常/总正常)]。
#通过上述的Logistic回归，剔除x1,x4,x6三个变量，对剩下的变量进行WOE转换。

#1、进行分箱  age变量(x2)：
cutx2= c(-Inf,30,35,40,45,50,55,60,65,75,Inf)
plot(cut(train$x2,cutx2))

#2,NumberOfTime30-59DaysPastDueNotWorse变量(x3)：
cutx3 = c(-Inf,0,1,3,5,Inf)
plot(cut(train$x3,cutx3))

#3,MonthlyIncome变量(x5)：
cutx5 = c(-Inf,1000,2000,3000,4000,5000,6000,7500,9500,12000,Inf)
plot(cut(train$x5,cutx5))

#4,NumberOfTimes90DaysLate变量(x7)：
cutx7 = c(-Inf,0,1,3,5,10,Inf)
plot(cut(train$x7,cutx7))

#5,NumberRealEstateLoansOrLines变量(x8)：
cutx8= c(-Inf,0,1,2,3,5,Inf)
plot(cut(train$x8,cutx8))

#6,NumberOfTime60-89DaysPastDueNotWorse变量(x9)：
cutx9 = c(-Inf,0,1,3,5,Inf)
plot(cut(train$x9,cutx9))

#7,NumberOfDependents变量(x10)：
cutx10 = c(-Inf,0,1,2,3,5,Inf)
plot(cut(train$x10,cutx10))


#七、计算WOE值计算WOE的函数
totalgood = as.numeric(table(train$y))[1]
totalbad = as.numeric(table(train$y))[2]
getWOE <- function(a,p,q)
{
  Good <- as.numeric(table(train$y[a > p & a <= q]))[1]
  Bad <- as.numeric(table(train$y[a > p & a <= q]))[2]
  WOE <- log((Bad/totalbad)/(Good/totalgood),base = exp(1))
  return(WOE)
}

#age.WOE
Agelessthan30.WOE=getWOE(train$x2,-Inf,30)
Age30to35.WOE=getWOE(train$x2,30,35)
Age35to40.WOE=getWOE(train$x2,35,40)
Age40to45.WOE=getWOE(train$x2,40,45)
Age45to50.WOE=getWOE(train$x2,45,50)
Age50to55.WOE=getWOE(train$x2,50,55)
Age55to60.WOE=getWOE(train$x2,55,60)
Age60to65.WOE=getWOE(train$x2,60,65)
Age65to75.WOE=getWOE(train$x2,65,75)
Agemorethan.WOE=getWOE(train$x2,75,Inf)
age.WOE=c(Agelessthan30.WOE,Age30to35.WOE,Age35to40.WOE,Age40to45.WOE,Age45to50.WOE,
          Age50to55.WOE,Age55to60.WOE,Age60to65.WOE,Age65to75.WOE,Agemorethan.WOE)
age.WOE

PastDuelessthan0.WOE=getWOE(train$x3,-Inf,0)
PastDue0to1.WOE=getWOE(train$x3,0,1)
PastDue1to3.WOE=getWOE(train$x3,1,3)
PastDue3to5.WOE=getWOE(train$x3,3,5)
PastDuemorethan.WOE=getWOE(train$x3,5,Inf)
PastDue.WOE=c(PastDuelessthan0.WOE,PastDue0to1.WOE,
              PastDue1to3.WOE,PastDue3to5.WOE,
              PastDuemorethan.WOE)
PastDue.WOE

MonthIncomelessthan1000.WOE=getWOE(train$x5,-Inf,1000)
MonthIncome1000to2000.WOE=getWOE(train$x5,1000,2000)
MonthIncome2000to3000.WOE=getWOE(train$x5,2000,3000)
MonthIncome3000to4000.WOE=getWOE(train$x5,3000,4000)
MonthIncome4000to5000.WOE=getWOE(train$x5,4000,5000)
MonthIncome5000to6000.WOE=getWOE(train$x5,5000,6000)
MonthIncome6000to7500.WOE=getWOE(train$x5,6000,7500)
MonthIncome7500to9500.WOE=getWOE(train$x5,7500,9500)
MonthIncome9500to12000.WOE=getWOE(train$x5,9500,12000)
MonthIncomemorethan.WOE=getWOE(train$x5,12000,Inf)
MonthIncome.WOE=c(MonthIncomelessthan1000.WOE,MonthIncome1000to2000.WOE,
                  MonthIncome2000to3000.WOE,MonthIncome3000to4000.WOE,
                  MonthIncome4000to5000.WOE,MonthIncome5000to6000.WOE,
                  MonthIncome6000to7500.WOE,MonthIncome7500to9500.WOE,
                  MonthIncome9500to12000.WOE,MonthIncomemorethan.WOE)
MonthIncome.WOE
#NumberOfTime90DaysPastDueNotWorse变量(x7)
#NumberOfTime90DaysPastDueNotWorse变量(x7)
Days90PastDuelessthan0.WOE = getWOE(train$x7,-Inf,0)
Days90PastDue0to1.WOE=getWOE(train$x7,0,1)
Days90PastDue1to3.WOE=getWOE(train$x7,1,3)
Days90PastDue3to5.WOE=getWOE(train$x7,3,5)
Days90PastDue5to10.WOE=getWOE(train$x7,5,10)
Days90sPastDuemorethan.WOE=getWOE(train$x7,10,Inf)
Days90sPastDue.WOE=c(Days90PastDuelessthan0.WOE,Days90PastDue0to1.WOE,
                     Days90PastDue1to3.WOE,Days90PastDue3to5.WOE,
                     Days90PastDue5to10.WOE,Days90sPastDuemorethan.WOE)
Days90sPastDue.WOE
RealEstatelessthan0.WOE = getWOE(train$x8,-Inf,0)
RealEstate0to1.WOE=getWOE(train$x8,0,1)
RealEstate1to2.WOE=getWOE(train$x8,1,2)
RealEstate2to3.WOE=getWOE(train$x8,2,3)
RealEstate3to5.WOE=getWOE(train$x8,3,5)
RealEstatemorethan.WOE=getWOE(train$x8,5,Inf)
RealEstate.WOE=c(RealEstatelessthan0.WOE,RealEstate0to1.WOE,
                 RealEstate1to2.WOE,RealEstate2to3.WOE,
                 RealEstate3to5.WOE,RealEstatemorethan.WOE)
RealEstate.WOE
Days60.89PastDuelessthan0.WOE = getWOE(train$x9,-Inf,0)
Days60.89PastDue0to1.WOE=getWOE(train$x9,0,1)
Days60.89PastDue1to3.WOE=getWOE(train$x9,1,3)
Days60.89PastDue3to5.WOE=getWOE(train$x9,3,5)
Days60.89PastDuemorethan.WOE=getWOE(train$x9,5,Inf)
Days60.89PastDue.WOE=c(Days60.89PastDuelessthan0.WOE,Days60.89PastDue0to1.WOE,
                       Days60.89PastDue1to3.WOE,Days60.89PastDue3to5.WOE,
                       Days60.89PastDuemorethan.WOE)
Days60.89PastDue.WOE
Dependentslessthan0.WOE = getWOE(train$x10,-Inf,0)
Dependents0to1.WOE=getWOE(train$x10,0,1)
Dependents1to2.WOE=getWOE(train$x10,1,2)
Dependents2to3.WOE=getWOE(train$x10,2,3)
Dependents3to5.WOE=getWOE(train$x10,3,5)
Dependentsmorethan.WOE=getWOE(train$x10,5,Inf)
Dependents.WOE=c(Dependentslessthan0.WOE,Dependents0to1.WOE,
                 Dependents1to2.WOE,Dependents2to3.WOE,
                 Dependents3to5.WOE,Dependentsmorethan.WOE)
Dependents.WOE


trainWOE<-train
trainWOE$y = 1-train$y
glm.fit = glm(y~x2+x3+x5+x7+x8+x9+x10,data = trainWOE,
              family = binomial(link = logit))
summary(glm.fit)
coe = (glm.fit$coefficients)
coe
factor <- 20/(log(30,base = 10)-log(15,base = 10))
offset <- 600-factor*log(15,base = 10)
a<-log(totalgood/totalbad,base = 10)
baseScore <- a*factor+offset
source('/home/jenyou/Credit_score/variableScoreCal.R', encoding = 'UTF-8')

getscore<-function(i,x){
  score = round(factor*as.numeric(coe[i])*x,0)
  return(score)
}
source('/home/jenyou/Credit_score/variableScoreCal.R', encoding = 'UTF-8')
baseScore
source('/home/jenyou/Credit_score/autoCalScore.R', encoding = 'UTF-8')
View(train)
