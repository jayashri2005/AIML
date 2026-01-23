"""
linear combination avg of all
shrink from -inf to +inf
-1 to +1 tanh function
0 to 1 sigma function
sigma=(e^x/1+e^2) when e=0 res=0.5 pow=minus = small value in 0.____
0 to 1 transform into probostic val range of 0 to 1 

activation fun -> we can transform it into diff 1 val into double(val), another
fun can transfor, 1 val into sigma val 0 to 1 used in logistic regression 

ml -> linear-> activation fun -> output[probostic val->0 to 1] -> classification -> less 0.5 erro 0.9 match more awway it match 

classification-> val greater than 0.5 
w change slope falt b increase curve sliding ref[hyperbolic tangent]
0 projected as 0.5 in the sigmoid 
loss is like s feedback to change w,b 

high,loss val diff -> way to compare -> suming -> optimse by keeping sum of error 
sumazion of all these num should be max 


->>>>in linear reg outliers[isolated  points] pull the line 
hugher error line more push towards its
p(h)>0.5=high
p(h)<0.5=low
this the reason for using sigmod function
adv -> big val of x -> curve not get distrub more it moves only slightly 

formula sigmoid = (e^x/1+e^x)

--->how loss calculated 
p(l)=1-p(h)

if we are having two labels for then consider 1 as 0 and aotherr as 1 9map like a x,y,z axis) then we look through this it like a two cluster near together so instead of using 
sigmoid we draw a line to classify them 
best fit line is the line that seperate cluster 

in mathetically we can cal loss by using the perpendicular distance from the line to the point 
>> when we transform sigmoid to line line would be 0 then we clasiify by using -1 to +1 
 /// probality dis yp = dist b/w line nd cluster [ sum of the dis = max to get optimised line ]
 sum of 0 and 1 should equal to max
 bigger the dis = val great 
 fix as -ve nd =ve the cal dis feom lines

 sum(+ve nd -ve) = max
 but it result in zero so we multiply +ve*+1 nd -ve as-ve*-1

 -->>>>>i.e sum(y)*sum(y^p)=max

 ipo sum panum pothu mult pana aprm ellam =ve na ok suppose -ve vantha athu max illa
 when misclassification takes place max meaning optimization doesnt takes place 

-->> sum(y)*sum(y^p) = max
if y is -ve the 1-y
simillary y^p is +ve then 1-y^p
>>>>>>> y.y^p+(1-y)(1-y^p)= max = (sum(y.y^p))

!_>>>>>> comdition 

1/n(sum[ y.y^p compliment (1-y)(1-y^p)])
loss= -optimization =[-1/n(sum[ y.y^p compliment (1-y)(1-y^p)])]

>> due to conflict we use log method

sigmoid = e^x/1+e^x 
log(sigmoid) = log(e^x/1+e^x) = x-log(1+e^x)

>>> log(e^z)=z [odds] 
z=liner comb(w1x1+w2x2+..)means a straigh tline 
we got crt loss nd high with the help of log fun 

y*log(y^p)+(1-y)*log(1-y^p) val = -ve  + -ve 
 
 ----> cal loss = -(-val + -val2) = val + val2
 loss = -ve log likelihood loss

 regression = loss= mse = avg err square

 loss cmpw with act vs prob 

good opti line nd cluster dis large 
why dist long bcoz upon seeing client side data it cause outlier or misclassification so if line is away it doesnt get misclassify
this is called "classification" used only catagorical data [rats,cats]


"""