import pprint
import os
import math
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import warnings
import collections as ct
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian
from pylab import plot, show,hist,figure,title
from fitter import Fitter
import statsmodels.api as sm

image_output = "../images/"

#Variables
Annual_Miles_Avg = 15000
Time_Worth=14.0
Car_Price=15001

def avgMPH(annualMiles):
  if annualMiles/365>100:
    mph = 60
  elif annualMiles/365>50:
    mph = 40
  elif annualMiles/365>30:
    mph = 35
  elif annualMiles/365>20:
    mph = 30
  else:
    mph = 25
  return mph

#EconomicParameters
Model_Length = 20 # as years
MPG = 25
Fuel_Price = 2.5  # as $/gal
#Trip_Time_Avg = 10  # as num
Num_Trips_Avg = 3  # as num

MPG = float(MPG)
Fuel_Price = float(Fuel_Price)

Num_Trips_Avg = float(Num_Trips_Avg)

#Global Inputs, Calculated
Daily_Miles_Avg = Annual_Miles_Avg/365
Trip_Dist_Avg = Daily_Miles_Avg/Num_Trips_Avg
Trip_Time_Avg = Daily_Miles_Avg/avgMPH(Annual_Miles_Avg)/Num_Trips_Avg*60 #as minutes
#Daily_Miles_Avg = [x / 365 for x in Annual_Miles_Avg]   # as miles/day
#Trip_Dist_Avg = [x / Num_Trips_Avg for x in Daily_Miles_Avg]
#Drive_Time_Avg = Trip_Time_Avg*Num_Trips_Avg   # as minutes

Trip_Time_Avg = float(Trip_Time_Avg)

######################################################################################################################
#Traditional Car Ownership Model
#Expenses to Purchase Car, Hardcoded
#Car_Price = 24000 #as $
Sales_Tax_Percent = .0625 #as decimal percent
Purchase_Fees = 250  # as $ for maybe title transfer
Down_Payment = 5000 #as $
Loan_Terms = 48 #as months
AIR = .03 #as decimal percent

Sales_Tax_Percent = float(Sales_Tax_Percent)
Purchase_Fees = float(Purchase_Fees)
Down_Payment = float(Down_Payment)
Loan_Terms = float(Loan_Terms)
AIR = float(AIR)
#Expenses to Purchase Car, Calculated
#Sales_Tax_Num = Car_Price*Sales_Tax_Percent #as $
#Loan_Amount = Car_Price - Down_Payment #as $
#used in car_ownership now
MIR = AIR/12
#Monthly_PMT = (MIR*Loan_Amount*(1+MIR)**Loan_Terms)/(((1+MIR)**Loan_Terms)-1)
#included in Car_Ownership Model

##################################################################################################################
#Direct Expenses to Own Car Avg'd over 10 years
Maint_Repairs_Exp = 1250 # as $
Insurance_Exp = 1000 # as $
Registr_Taxes_Exp = 150 # as $
Parking_Exp = 200 # as $
Maint_Repairs_Exp = float(Maint_Repairs_Exp)
Insurance_Exp = float(Insurance_Exp)
Registr_Taxes_Exp = float(Registr_Taxes_Exp)
Parking_Exp = float(Parking_Exp)


#Now in Car_Ownership
#Fuel_Exp = [Annual_Miles_Avg[i]/MPG[i]*Fuel_Price[i] for i in range(0,simsize-1)]
#Total_DE = Maint_Repairs_Exp+Insurance_Exp+Registr_Taxes_Exp+Parking_Exp+Fuel_Exp

#####################################################################################################################
#-----------Uber Model
#-----Expenses for Hiring
Fare_Base = 2 # as $
Fare_Per_Mile = 1.50 # as $/mile
Fare_Per_Minute = .20 # as $/minute
#In Uber_Model

#In Uber_Model
#Uber_Exp_Annual = Uber_Exp_Daily*365
Min_Fee_Uber = 6.0 # as $
Cancel_Fee_Uber = 8.0 # as $
Fare_Base = float(Fare_Base)
Fare_Per_Mile = float(Fare_Per_Mile)
Fare_Per_Minute = float(Fare_Per_Minute)

Uber_Trip_Avg = Fare_Base+Fare_Per_Mile*Trip_Dist_Avg+Fare_Per_Minute*Trip_Time_Avg
Uber_Trip_Avg = float(Uber_Trip_Avg)
Uber_Exp_Daily =  Uber_Trip_Avg*Num_Trips_Avg

#Uber_Exp_Annual = float(Uber_Exp_Annual)
Min_Fee_Uber = float(Min_Fee_Uber)
Cancel_Fee_Uber = float(Cancel_Fee_Uber)

#---Annual Value of Time
Wait_Time_Uber_Avg = 5.0 # as minutes
Wait_Time_Uber_Avg = float(Wait_Time_Uber_Avg)
#Add to Uber Model
#Val_Time_Daily_Uber = (Wait_Time_Uber_Avg*Num_Trips_Avg)/60*Time_Worth
#Val_Time_Annual_Uber = Val_Time_Daily_Uber*365
###############################################################################################################
#Annual Indirect Expenses Avg'd over 10 years
Property_Tax_Garage_IDE= 300 # as $ Garage might be worth $15k; if property tax 2%; garage $300/year
Garage_Repair_IDE_Ann = 200 # as $
Property_Tax_Garage_IDE = float(Property_Tax_Garage_IDE)
Garage_Repair_IDE_Ann = float(Garage_Repair_IDE_Ann)
Total_IDE = Property_Tax_Garage_IDE+Garage_Repair_IDE_Ann

#Annual Value of Time
Walk_Time_Avg= 20 # as minutes
Walk_Time_Avg = float(Walk_Time_Avg)
#Val_Drive_Time_Daily = Time_Worth*Drive_Time_Avg/60
#Val_Walk_Time_Daily = Walk_Time_Avg/60*Time_Worth
#Val_Time_Daily_Car =Val_Drive_Time_Daily+Val_Walk_Time_Daily
#Val_Time_Annual_Car = (Val_Time_Daily_Car)*365
#used in CarOwnership Function now
#---------------------------------------------------------------------------

#CELL

#####################################################################################################################
def Amortization(mnth = 12,loanterms= Loan_Terms, ir = MIR,carprice= Car_Price, downpmnt = Down_Payment):
    loan = carprice-downpmnt #as $
    pmnt = (ir*loan*(1+ir)**loanterms)/(((1+ir)**loanterms)-1)

    #Year1 Values
    Interest_1 = loan*ir
    #print(Interest_1)
    Principal_1 = pmnt-Interest_1
    #print(Principal_1)
    Balance_1 = loan-Principal_1
    Interest = ir*Balance_1
    Balance = Balance_1
    Total_Principal = Principal_1
    Total_Interest = Interest_1
    Total_Pmnt = pmnt*mnth
    #print(Total_Pmnt)
    for i in range(0,int(mnth-1)):
        Interest = ir*Balance
        Principal = pmnt-Interest
        Balance = Balance-Principal
        Total_Principal = Principal+Total_Principal
        Total_Interest = Interest +Total_Interest
    if mnth==1:
        return pmnt, pmnt, Interest_1, Interest_1, Principal_1, Principal_1, Balance_1
    else:
        return pmnt,Total_Pmnt,Interest,Total_Interest,Principal,Total_Principal,Balance

def Uber_Expense_Model(outputyears = (7,10,15,20), annualroi = 5,
                       farebase = Fare_Base, farepermile = Fare_Per_Mile, farepermin = Fare_Per_Minute,
                       annualmiles = Annual_Miles_Avg, triptime = Trip_Time_Avg, numtrips = Num_Trips_Avg,
                       timeworth= Time_Worth, waittime = Wait_Time_Uber_Avg, carprice = Car_Price):

    #Calc Val_Time_Annual_Uber
    Val_Time_Daily_Uber = (waittime*numtrips)/60*timeworth
    valuetime = Val_Time_Daily_Uber*365

    #Calc Trip_Dist_Avg
    Daily_Miles_Avg = annualmiles/365   # as miles/day
    Trip_Dist_Avg = Daily_Miles_Avg/numtrips

    #Calc Uber_Exp_Annual
    Uber_Trip_Avg = farebase+farepermile*Trip_Dist_Avg+farepermin*triptime
    Uber_Exp_Daily =  Uber_Trip_Avg*Num_Trips_Avg
    annualexpense = Uber_Exp_Daily*365

    a,OUP_Expense_Car,Cum_Expense_Car,b = Car_Ownership_Expense_Model(annualmiles=annualmiles,timeworth=timeworth, carprice= carprice)
    #Convert annualroi to float
    annualroi = float(annualroi)

    array = np.zeros((10,len(OUP_Expense_Car)))
    #Annual Fee to TNC
    array[0] = annualexpense
    #Value of Time Expense
    array[1] = valuetime
    #Total OUP Expense
    array[2] = array[0] + array[1]
    #Cum. OOP Expense
    for i in range(0,len(OUP_Expense_Car)):
        if i==0:
            array[3,i]=array[2,i]
        else:
            array[3,i]=array[3,i-1]+array[2,i]
    #Unrealized Annual Cash Savings
    array[4] = array[2] - OUP_Expense_Car
    #Cum Unrealized Cash Savings
    for i in range(0,len(OUP_Expense_Car)):
        if i==0:
            array[5,i]=array[4,i]
        else:
            array[5,i]=array[5,i-1]+array[4,i]
    ##Unrealized Value of Invested Savings
    for i in range(0,len(OUP_Expense_Car)):
        if i==0:
            array[6,i]=array[4,i]
        else:
            array[6,i]=array[4,i]+(array[6,i-1]*(1+annualroi/100))
    #Unrealized Added Value of Investing
    array[7] = array[6]-array[5]

    #Total Expense for Using Uber
    array[8] = array[3]+array[7]

    #Expense Delta
    array[9] = Cum_Expense_Car-array[8]

    #output array
    output = np.zeros((1,len(outputyears)))
    for i in range(0,len(outputyears)):
        output[0,i] = array[8,outputyears[i]-1]

    #Add list together to get matrix
    matrix = np.array([array[0],array[1],array[2],array[3],array[4],array[5],array[6],array[7],array[8],array[9]])
    return output,matrix

def Car_Ownership_Expense_Model(outputyears = (7,10,15,20), n = [7,10,15,20], purchasefees = Purchase_Fees,
                                 downpmnt = Down_Payment,ide = Total_IDE, timeworth= Time_Worth,
                                 annualmiles = Annual_Miles_Avg, loanterms = Loan_Terms,
     years= Model_Length,carprice = Car_Price, milespergallon = MPG, triptime = Trip_Time_Avg,
         fuelprice = Fuel_Price, numtrips = Num_Trips_Avg, taxpercent = Sales_Tax_Percent,
         maintenance = Maint_Repairs_Exp, insurance = Insurance_Exp,registrationtaxes = Registr_Taxes_Exp,
        parking = Parking_Exp):


    Fuel_Exp = annualmiles/milespergallon*fuelprice
    de = maintenance+insurance+registrationtaxes+parking+Fuel_Exp


    salestax = carprice*taxpercent #as $
    Loan_Amount = carprice - Down_Payment #as $


    Daily_Miles_Avg = annualmiles/365   # as miles/day
    Trip_Dist_Avg = Daily_Miles_Avg/numtrips
    Drive_Time_Avg = triptime*numtrips   # as minutes

    #timeworth is in here
    Val_Drive_Time_Daily = timeworth*Drive_Time_Avg/60
    Val_Walk_Time_Daily = Walk_Time_Avg/60*timeworth
    Val_Time_Daily_Car =Val_Drive_Time_Daily+Val_Walk_Time_Daily
    Val_Time_Annual_Car = (Val_Time_Daily_Car)*365
    valuetime = Val_Time_Annual_Car


    loanyears = loanterms/12
    array = np.zeros((13,years))
    [rows,columns] = array.shape
    [a,b,c,d,e,f,g] = Amortization(mnth=12,carprice=carprice)
    Principal_Incr = f
    Interest_Incr = d

    #assigning Cost at Time of Purchase
    cost = salestax+purchasefees+downpmnt
    array[0,0] = cost
    #Principal Paid and Interest Paid
    for i in range(0,columns):
        if i > loanyears-1:
            array[1,i] = 0.0
            array[2,i] = 0.0
        elif i == 0:
            array[1,i] = Principal_Incr
            array[2,i] = Interest_Incr
        else:
            [a,b,c,d,e,f,g] = Amortization(mnth=(i+1)*12,carprice=carprice)
            [h,j,k,l,m,n,o] = Amortization(mnth = i*12,carprice=carprice)
            Principal_Incr = f-n
            Interest_Incr = d-l
            array[1,i] = Principal_Incr
            array[2,i] = Interest_Incr

    #Assigning DE and IDE
    array[3] = de
    array[4] = ide

    #Summing expense Total Value Out of Pocket Expense and Cumulative Out of Pocket Expense
    array[5] = array[0]+array[1]+array[2]+array[3]+array[4]
    for i in range(0,columns):
        if i==0:
            array[6,i]=array[5,i]
        else:
            array[6,i]=array[5,i]+array[6,i-1]

    #Annual Value of Time and Cumulative Value of Time
    array[7] = valuetime
    for i in range(0,columns):
        if i==0:
            array[8,i]=array[7,i]
        else:
            array[8,i]=array[7,i]+array[8,i-1]

    #Total Annual Expense for Owning and Cum. Expense for Owning
    array[9] = array[7] +array[5]
    for i in range(0,columns):
        if i==0:
            array[10,i]=array[9,i]
        else:
            array[10,i]=array[9,i]+array[10,i-1]

    #Value of Car
    array[11] = Depreciation(20,4)*carprice

    #Cumulative Expenses Less value of Car
    array[12] = array[10] - array[11]

    output = np.zeros((2,len(outputyears)))
    for i in range(0,len(outputyears)):
        output[0,i] = array[10,outputyears[i]-1]
        output[1,i] = array[12,outputyears[i]-1]

    matrix = np.array([array[0],array[1],array[2],array[3],array[4],array[5],array[6],array[7],array[8],array[9],array[10],array[11],array[12]])
    return output,array[5],array[10],matrix

def Depreciation(initialrate,rate,years = Model_Length):
    years = int(math.ceil(years))
    array = np.zeros((1,years))
    initialrate = initialrate/100.0
    rate = rate/100.0
    totalpercent = 1.0
    for i in range(0,(array.size)):
        if initialrate>.1:
            array[0][i] = totalpercent*(1-initialrate)
            initialrate = initialrate - rate
            totalpercent= array[0][i]
        else:
            array[0][i] = totalpercent*(1-.1)
            totalpercent= array[0][i]
    return array


#Single Test Case used to verify model matches excel document
output_year = (7,)

a,b,c,d = Car_Ownership_Expense_Model(outputyears= output_year,annualmiles = Annual_Miles_Avg,carprice=Car_Price,timeworth= Time_Worth)
d = Uber_Expense_Model(outputyears= output_year,timeworth=Time_Worth,annualmiles=Annual_Miles_Avg,carprice = Car_Price)

####################
#Traditional Car Ownership Model
#Expenses to Purchase Car, Hardcoded
#Car_Price = 24000 #as $
Sales_Tax_Percent = .0625 #as decimal percent
Purchase_Fees = 250  # as $ for maybe title transfer
Down_Payment = 5000 #as $
Loan_Terms = 48 #as months
AIR = .03 #as decimal percent

Sales_Tax_Percent = float(Sales_Tax_Percent)
Purchase_Fees = float(Purchase_Fees)
Down_Payment = float(Down_Payment)
Loan_Terms = float(Loan_Terms)
AIR = float(AIR)
#Expenses to Purchase Car, Calculated
#Sales_Tax_Num = Car_Price*Sales_Tax_Percent #as $
#Loan_Amount = Car_Price - Down_Payment #as $
#used in car_ownership now
MIR = AIR/12
#Monthly_PMT = (MIR*Loan_Amount*(1+MIR)**Loan_Terms)/(((1+MIR)**Loan_Terms)-1)
#included in Car_Ownership Model

##################################################################################################################
#Direct Expenses to Own Car Avg'd over 10 years
Maint_Repairs_Exp = 1250 # as $
Insurance_Exp = 1000 # as $
Registr_Taxes_Exp = 150 # as $
Parking_Exp = 200 # as $
Maint_Repairs_Exp = float(Maint_Repairs_Exp)
Insurance_Exp = float(Insurance_Exp)
Registr_Taxes_Exp = float(Registr_Taxes_Exp)
Parking_Exp = float(Parking_Exp)

#Now in Car_Ownership
#Fuel_Exp = [Annual_Miles_Avg[i]/MPG[i]*Fuel_Price[i] for i in range(0,simsize-1)]
#Total_DE = Maint_Repairs_Exp+Insurance_Exp+Registr_Taxes_Exp+Parking_Exp+Fuel_Exp

#####################################################################################################################
#-----------Uber Model
#-----Expenses for Hiring
Fare_Base = 2 # as $
Fare_Per_Mile = 1.50 # as $/mile
Fare_Per_Minute = .20 # as $/minute
#In Uber_Model

#In Uber_Model
#Uber_Exp_Annual = Uber_Exp_Daily*365
Min_Fee_Uber = 6.0 # as $
Cancel_Fee_Uber = 8.0 # as $
Fare_Base = float(Fare_Base)
Fare_Per_Mile = float(Fare_Per_Mile)
Fare_Per_Minute = float(Fare_Per_Minute)

#Uber_Trip_Avg = Fare_Base+Fare_Per_Mile*Trip_Dist_Avg+Fare_Per_Minute*Trip_Time_Avg
#Uber_Trip_Avg = float(Uber_Trip_Avg)
#Uber_Exp_Daily =  Uber_Trip_Avg*Num_Trips_Avg


#Uber_Exp_Annual = float(Uber_Exp_Annual)
Min_Fee_Uber = float(Min_Fee_Uber)
Cancel_Fee_Uber = float(Cancel_Fee_Uber)

#---Annual Value of Time
Wait_Time_Uber_Avg = 5.0 # as minutes
Wait_Time_Uber_Avg = float(Wait_Time_Uber_Avg)
#Add to Uber Model
#Val_Time_Daily_Uber = (Wait_Time_Uber_Avg*Num_Trips_Avg)/60*Time_Worth
#Val_Time_Annual_Uber = Val_Time_Daily_Uber*365
###############################################################################################################
#Annual Indirect Expenses Avg'd over 10 years
Property_Tax_Garage_IDE= 300 # as $ Garage might be worth $15k; if property tax 2%; garage $300/year
Garage_Repair_IDE_Ann = 200 # as $
Property_Tax_Garage_IDE = float(Property_Tax_Garage_IDE)
Garage_Repair_IDE_Ann = float(Garage_Repair_IDE_Ann)
Total_IDE = Property_Tax_Garage_IDE+Garage_Repair_IDE_Ann

#Annual Value of Time
Walk_Time_Avg= 20 # as minutes
Walk_Time_Avg = float(Walk_Time_Avg)
#Val_Drive_Time_Daily = Time_Worth*Drive_Time_Avg/60
#Val_Walk_Time_Daily = Walk_Time_Avg/60*Time_Worth
#Val_Time_Daily_Car =Val_Drive_Time_Daily+Val_Walk_Time_Daily
#Val_Time_Annual_Car = (Val_Time_Daily_Car)*365
#used in CarOwnership Function now

def carpooling(ubermodel,carmodel,timeworth,carprice,densityfactor,annualmiles=Annual_Miles_Avg,numtrips=Num_Trips_Avg,farebase=Fare_Base,farepermile=Fare_Per_Mile,farepermin=Fare_Per_Minute):

#Find Avg Uber Trip Cost
  Daily_Miles_Avg = annualmiles/365
  Trip_Dist_Avg = Daily_Miles_Avg/numtrips
  Trip_Time_Avg = Daily_Miles_Avg/avgMPH(annualmiles)/numtrips*60
  Uber_Trip_Avg = Fare_Base+Fare_Per_Mile*Trip_Dist_Avg+Fare_Per_Minute*Trip_Time_Avg

#Find Allowable Charge Per Trip
  array = np.zeros((9,carmodel.shape[1]))
  array[0]=carmodel[0]+carmodel[1]
  array[1]=carmodel[2]+carmodel[3]
  array[2]=carmodel[4]
  array[3]=carmodel[7]
  array[4]=ubermodel[1]
  array[5]=carmodel[11]
  array[6]=ubermodel[7]
  for i in range(array.shape[1]):
    array[7][i]=(sum(array[0][0:(i+1)])+sum(array[1][0:(i+1)])+sum(array[2][0:(i+1)]))+sum(array[3][0:(i+1)])-sum(array[4][0:(i+1)])-array[5][i]-array[6][i]
  years = np.arange(1,carmodel.shape[1]+1)
  array[8]=array[7]/(numtrips*365*years)

#Create grid for heatmap
  hm=np.zeros((len(densityfactor),array.shape[1]))
  #This can probably be done with numpy operations better
  for r in range(hm.shape[0]):
    for c in range(hm.shape[1]):
      hm[r][c]=densityfactor[r]*array[8][c]-Uber_Trip_Avg

#Actual heatmap
  # plt.figure(figsize=(9,9))
  plt.title("Annual Miles = {:,} , Dollar Per Mile = ${:.2f}\nTime Worth Per Hour = ${:.2f} , Car Price = \${:,}".format(10000,farepermile,timeworth,carprice), size = 10)
  # plt.title('Annual Miles = {:,}'.format(annualmiles) + ' , Dollar Per Mile = ${1:.2f}'.format(farepermile)+'\n'+input_text, size = 10)
  ax = sns.heatmap(hm[:,0:15], vmin=-25,vmax=25,cmap='RdBu',cbar_kws={"orientation":"horizontal"},linecolor='Black',cbar=False)
  # ax = sns.heatmap(hm, annot=False, fmt="d", square = False, cmap = 'RdBu', center = 0, linecolor = 'Black',yticklabels=10, xticklabels=10, cbar_kws={"orientation": "horizontal"},vmin = -150000, vmax = 150000, cbar = False)
  # ax.set_xticklabels(['${}'.format(int(i.get_text())/ 1000) for i in ax.get_xticklabels()])
  # ax.set_yticklabels(['${}'.format(int(i.get_text()) ) for i in ax.get_yticklabels()])
  # ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0)
  ax.set_xticklabels(years)
  ynums = np.linspace(1,3,21)
  ax.set_yticklabels(ynums,rotation=0)
  # test = ['1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0','2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9','3.0']
  # ax.set_yticklabels(test,rotation=0)
  # ax.set_yticklabels(densityfactor)
  # ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0)
  plt.rc('xtick', labelsize=25)
  plt.rc('ytick', labelsize=25)
  plt.xlabel('Years', size = 10 )
  plt.ylabel('Density Factor', size = 10)
  # plt.savefig(image_output+'Carpool Heat Map'+'.png', bbox_inches='tight')
  plt.savefig(image_output+'Carpool Heat Map.png')

def main():
  timeworth=14
  carprice=20000
  annualmiles=9855
  numtrips=3

  (car_output,a,b,car_matrix) = Car_Ownership_Expense_Model(timeworth = timeworth,carprice= carprice,annualmiles=annualmiles,numtrips=numtrips)
  (uber_output,uber_matrix) = Uber_Expense_Model(timeworth=timeworth,carprice=carprice,annualmiles=annualmiles,numtrips=numtrips,annualroi=2)
  carpooling(carprice=carprice,timeworth=timeworth,annualmiles=annualmiles,numtrips=3,ubermodel=uber_matrix,carmodel=car_matrix,densityfactor=np.arange(1,3.05,.05))

if __name__ == "__main__":
  main()
