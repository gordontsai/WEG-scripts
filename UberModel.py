import pprint  
import math
import numpy as np
import pdb
import random
import scipy.stats
#import seaborn as sns
#import matplotlib.pyplot as plt
#from Ipython.Debug import Tracer
#Uber Model based on xlsx spreadsheet
#Global Inputs, Hardcoded

print(scipy.stats.norm.pdf(0,1))

#Distributions
#Varied 
	#Time Worth - data for average salary distribution in different metro regions 
	#Car Price - hard to get more resolution
	#Average Annual Mile Driven 

#######################################


#EconomicParameters
Model_Length = 20 # as years


MPG = 42
Annual_Miles_Avg = 9000
Fuel_Price = 2.25  # as $/gal
Trip_Time_Avg = 15  # as num
Num_Trips_Avg = 4  # as num
Time_Worth = 100  # as $/hr
MPG = float(MPG)
Annual_Miles_Avg = float(Annual_Miles_Avg)
Fuel_Price = float(Fuel_Price)
Trip_Time_Avg = float(Trip_Time_Avg)
Num_Trips_Avg = float(Num_Trips_Avg)
Time_Worth = float(Time_Worth)

#Global Inputs, Calculated
Daily_Miles_Avg = Annual_Miles_Avg/365   # as miles/day
Trip_Dist_Avg = Daily_Miles_Avg/Num_Trips_Avg
Drive_Time_Avg = Trip_Time_Avg*Num_Trips_Avg   # as minutes

#Traditional Car Ownership Model 
#Expenses to Purchase Car, Hardcoded
Car_Price = 24000 #as $
Sales_Tax_Percent = .0625 #as decimal percent
Purchase_Fees = 250  # as $ for maybe title transfer
Down_Payment = 5000 #as $
Loan_Terms = 48 #as months
AIR = .03 #as decimal percent
Car_Price = float(Car_Price)
Sales_Tax_Percent = float(Sales_Tax_Percent)
Purchase_Fees = float(Purchase_Fees)
Down_Payment = float(Down_Payment)
Loan_Terms = float(Loan_Terms)
AIR = float(AIR)
#Expenses to Purchase Car, Calculated
Sales_Tax_Num = Car_Price*Sales_Tax_Percent #as $
Loan_Amount = Car_Price - Down_Payment #as $
MIR = AIR/12
Monthly_PMT = (MIR*Loan_Amount*(1+MIR)**Loan_Terms)/(((1+MIR)**Loan_Terms)-1)

#Direct Expenses to Own Car Avg'd over 10 years
Maint_Repairs_Exp = 1250 # as $
Insurance_Exp = 1000 # as $
Registr_Taxes_Exp = 150 # as $
Parking_Exp = 200 # as $
Maint_Repairs_Exp = float(Maint_Repairs_Exp)
Insurance_Exp = float(Insurance_Exp)
Registr_Taxes_Exp = float(Registr_Taxes_Exp)
Parking_Exp = float(Parking_Exp)
Fuel_Exp = Annual_Miles_Avg/MPG*Fuel_Price
Total_DE = Maint_Repairs_Exp+Insurance_Exp+Registr_Taxes_Exp+Parking_Exp+Fuel_Exp

#Annual Indirect Expenses Avg'd over 10 years
Property_Tax_Garage_IDE= 300 # as $ Garage might be worth $15k; if property tax 2%; garage $300/year
Garage_Repair_IDE_Ann = 200 # as $
Property_Tax_Garage_IDE = float(Property_Tax_Garage_IDE)
Garage_Repair_IDE_Ann = float(Garage_Repair_IDE_Ann)
Total_IDE = Property_Tax_Garage_IDE+Garage_Repair_IDE_Ann

#Annual Value of Time
Walk_Time_Avg= 20 # as minutes
Walk_Time_Avg = float(Walk_Time_Avg)
Val_Drive_Time_Daily = Time_Worth*Drive_Time_Avg/60
Val_Walk_Time_Daily = Walk_Time_Avg/60*Time_Worth
Val_Time_Daily_Car =Val_Drive_Time_Daily+Val_Walk_Time_Daily
Val_Time_Annual_Car = (Val_Time_Daily_Car)*365

#---------------------------------------------------------------------------

#-----------Uber Model
#-----Expenses for Hiring
Fare_Base = 2 # as $
Fare_Per_Mile = 1.50 # as $/mile
Fare_Per_Minute = .20 # as $/minute
Uber_Trip_Avg = Fare_Base+Fare_Per_Mile*Trip_Dist_Avg+Fare_Per_Minute*Trip_Time_Avg
Uber_Exp_Daily =  Uber_Trip_Avg*Num_Trips_Avg

Uber_Exp_Annual = Uber_Exp_Daily*365
Min_Fee_Uber = 6.0 # as $
Cancel_Fee_Uber = 8.0 # as $
Fare_Base = float(Fare_Base)
Fare_Per_Mile = float(Fare_Per_Mile)
Fare_Per_Minute = float(Fare_Per_Minute)
Uber_Trip_Avg = float(Uber_Trip_Avg)
Uber_Exp_Annual = float(Uber_Exp_Annual)
Min_Fee_Uber = float(Min_Fee_Uber)
Cancel_Fee_Uber = float(Cancel_Fee_Uber)

#---Annual Value of Time
Wait_Time_Uber_Avg = 5.0 # as minutes 
Wait_Time_Uber_Avg = float(Wait_Time_Uber_Avg)
Val_Time_Daily_Uber = (Wait_Time_Uber_Avg*Num_Trips_Avg)/60*Time_Worth
Val_Time_Annual_Uber = Val_Time_Daily_Uber*365

def Amortization(mnth = Loan_Terms,pmnt = Monthly_PMT, ir = MIR, loan = Loan_Amount): 
	#Year1 Values
	Interest_1 = loan*ir
	Principal_1 = pmnt-Interest_1
	Balance_1 = loan-Principal_1
	Interest = ir*Balance_1
	Balance = Balance_1
	Total_Principal = Principal_1
	Total_Interest = Interest_1
	Total_Pmnt = pmnt*mnth
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


def Depreciation(initialrate,rate,years = Model_Length):
	years = math.ceil(years)
	array = np.zeros((1,years))
	initialrate=initialrate/100.0
	rate=rate/100.0
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


def Car_Ownsership_Expense_Model(outputyears = (7,10,15,20), n = [7,10,15,20], salestax = Sales_Tax_Num,
 purchasefees = Purchase_Fees, downpmnt = Down_Payment,de = Total_DE,
 ide = Total_IDE, valuetime = Val_Time_Annual_Car, years= Model_Length, loanyears = Loan_Terms,carprice = Car_Price):
	loanyears = loanyears/12
	array = np.zeros((13,years))
	[rows,columns] = array.shape
	[a,b,c,d,e,f,g] = Amortization(mnth=12)
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
			[a,b,c,d,e,f,g] = Amortization(mnth=(i+1)*12)
			[h,j,k,l,m,n,o] = Amortization(mnth = i*12)
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

	return output, array[5], array[10]

#a = Car_Ownsership_Expense_Model()
#print(a)

def Uber_Expense_Model(outputyears = (7,10,15,20),annualexpense = Uber_Exp_Annual, valuetime = Val_Time_Annual_Uber, annualroi = 5):
	a, OUP_Expense_Car, Cum_Expense_Car = Car_Ownsership_Expense_Model()
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


	return output



a = Uber_Expense_Model()
print(a)







#Test
#[a,b,c,d,e,f,g] = Amortization(mnth=1)
#print(a,b,c,d,e,f,g)