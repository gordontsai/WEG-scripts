from helper import avgMPH


# Car Ownership Variables
Loan_Terms=48       # as months
AIR=.03             # Annual Interest Rate as %
MIR=AIR/12          # Monthly Interest Rate as %
Car_Price=24000     # Car Price as $
Down_Payment=5000   # as $
Property_Tax_Garage_IDE=300 # as $ - Garage might be work 15k, if prop tax is 2% then garage is ~ $300/yr
Garage_Repair_IDE_Ann = 200 # as $
Total_IDE = Property_Tax_Garage_IDE+Garage_Repair_IDE_Ann # as $
Walk_Time_Avg= 20   # as minutes
## Direct Expenses to own Car averaged over 10 years
Maint_Repairs_Exp=1250  # as $
Insurance_Exp = 1000    # as $
Registr_Taxes_Exp = 150 # as $
Parking_Exp = 200       # as $



# TNC Variables
Fare_Base=2         # as $
Fare_Per_Mile=1.5   # as $/mile
Fare_Per_Minute=.20 # as $/minute


Annual_Miles_Avg=13000 # as # of miles
Daily_Miles_Avg=Annual_Miles_Avg/365
Num_Trips_Avg=2        # as # - we played around with 3 as a default before
Trip_Time_Avg = Daily_Miles_Avg/avgMPH(Annual_Miles_Avg)/Num_Trips_Avg*60 # as minutes
Wait_Time_Uber_Avg=5   # as minutes
Purchase_Fees=250      # as $ - this includes stuff like title transfer


Time_Worth=55       # as $/hr

# Economic Variables
Model_Length=20      # as yrs
MPG= 25              # as miles per gallon
Fuel_Price=2.5       # as $/gallon
Sales_Tax_Percent=.0625 # as decimal or percent if * by 100
subsidy=0            # as $


#############################################################################################
#Temp code that was in the file

# simsize = 100000
# # Time_Worth = npr.normal(loc =150, scale = 30,size=simsize)
# # Car_Price = npr.normal(loc = 24000, scale = 2400,size=simsize)
# # Annual_Miles_Avg = npr.normal(loc = 9000, scale =900,size=simsize)

# Time_Worth = 55
# Car_Price = 50000
# Annual_Miles_Avg = 13000

# ##############################################################################################################
# #EconomicParameters
# Model_Length = 20 # as years
# MPG = 25
# Fuel_Price = 2.5  # as $/gal
# #Trip_Time_Avg = 10  # as num
# Num_Trips_Avg = 2  # as num

# MPG = float(MPG)
# Fuel_Price = float(Fuel_Price)

# Num_Trips_Avg = float(Num_Trips_Avg)

# if isinstance(Time_Worth, np.ndarray):
#     Time_Worth = [float(Time_Worth[i]) for i in range(0,Time_Worth.size-1)]
# else:
#Time_Worth = float(Time_Worth)

# if isinstance(Annual_Miles_Avg, np.ndarray):
#     Annual_Miles_Avg = [float(Annual_Miles_Avg[i]) for i in range(0,Annual_Miles_Avg.size-1)]
# else:
#Annual_Miles_Avg = float(Annual_Miles_Avg)

# if isinstance(Car_Price, np.ndarray):
#     Car_Price = [float(Car_Price[i]) for i in range(0,Car_Price.size-1)]
# else:
#Car_Price = float(Car_Price)

# #Global Inputs, Calculated
# Daily_Miles_Avg = Annual_Miles_Avg/365
# Trip_Dist_Avg = Daily_Miles_Avg/Num_Trips_Avg
# Trip_Time_Avg = Daily_Miles_Avg/avgMPH(Annual_Miles_Avg)/Num_Trips_Avg*60 #as minutes
# #Daily_Miles_Avg = [x / 365 for x in Annual_Miles_Avg]   # as miles/day
# #Trip_Dist_Avg = [x / Num_Trips_Avg for x in Daily_Miles_Avg]
# #Drive_Time_Avg = Trip_Time_Avg*Num_Trips_Avg   # as minutes

# Trip_Time_Avg = float(Trip_Time_Avg)

# ######################################################################################################################
# #Traditional Car Ownership Model
# #Expenses to Purchase Car, Hardcoded
# #Car_Price = 24000 #as $
# Sales_Tax_Percent = .0625 #as decimal percent
# Purchase_Fees = 250  # as $ for maybe title transfer
# Down_Payment = 5000 #as $
# Loan_Terms = 48 #as months
# AIR = .03 #as decimal percent

# Sales_Tax_Percent = float(Sales_Tax_Percent)
# Purchase_Fees = float(Purchase_Fees)
# Down_Payment = float(Down_Payment)
# Loan_Terms = float(Loan_Terms)
# AIR = float(AIR)
# #Expenses to Purchase Car, Calculated
# #Sales_Tax_Num = Car_Price*Sales_Tax_Percent #as $
# #Loan_Amount = Car_Price - Down_Payment #as $
# #used in car_ownership now
# MIR = AIR/12
# #Monthly_PMT = (MIR*Loan_Amount*(1+MIR)**Loan_Terms)/(((1+MIR)**Loan_Terms)-1)
# #included in Car_Ownership Model

# ##################################################################################################################
# #Direct Expenses to Own Car Avg'd over 10 years
# Maint_Repairs_Exp = 1250 # as $
# Insurance_Exp = 1000 # as $
# Registr_Taxes_Exp = 150 # as $
# Parking_Exp = 200 # as $
# Maint_Repairs_Exp = float(Maint_Repairs_Exp)
# Insurance_Exp = float(Insurance_Exp)
# Registr_Taxes_Exp = float(Registr_Taxes_Exp)
# Parking_Exp = float(Parking_Exp)


# #Now in Car_Ownership
# #Fuel_Exp = [Annual_Miles_Avg[i]/MPG[i]*Fuel_Price[i] for i in range(0,simsize-1)]
# #Total_DE = Maint_Repairs_Exp+Insurance_Exp+Registr_Taxes_Exp+Parking_Exp+Fuel_Exp

# #####################################################################################################################
# #-----------Uber Model
# #-----Expenses for Hiring
# Fare_Base = 2 # as $
# Fare_Per_Mile = 1.50 # as $/mile
# Fare_Per_Minute = .20 # as $/minute
# # Fare_Base = 1 # as $
# # Fare_Per_Mile = .750 # as $/mile
# # Fare_Per_Minute = .10 # as $/minute
# #In Uber_Model

# #In Uber_Model
# #Uber_Exp_Annual = Uber_Exp_Daily*365
# Min_Fee_Uber = 6.0 # as $
# Cancel_Fee_Uber = 8.0 # as $
# Fare_Base = float(Fare_Base)
# Fare_Per_Mile = float(Fare_Per_Mile)
# Fare_Per_Minute = float(Fare_Per_Minute)

# Uber_Trip_Avg = Fare_Base+Fare_Per_Mile*Trip_Dist_Avg+Fare_Per_Minute*Trip_Time_Avg
# Uber_Trip_Avg = float(Uber_Trip_Avg)
# Uber_Exp_Daily =  Uber_Trip_Avg*Num_Trips_Avg

# #Uber_Exp_Annual = float(Uber_Exp_Annual)
# Min_Fee_Uber = float(Min_Fee_Uber)
# Cancel_Fee_Uber = float(Cancel_Fee_Uber)

# #---Annual Value of Time
# Wait_Time_Uber_Avg = 5.0 # as minutes
# Wait_Time_Uber_Avg = float(Wait_Time_Uber_Avg)
# #Add to Uber Model
# #Val_Time_Daily_Uber = (Wait_Time_Uber_Avg


# #EconomicParameters
# Model_Length = 20 # as years
# MPG = 25
# Fuel_Price = 2.5  # as $/gal
# #Trip_Time_Avg = 10  # as num
# Num_Trips_Avg = 2  # as num #############Change this back to 3 later

# MPG = float(MPG)
# Fuel_Price = float(Fuel_Price)
# #Trip_Time_Avg = float(Trip_Time_Avg)
# Num_Trips_Avg = float(Num_Trips_Avg)

# if isinstance(Time_Worth, np.ndarray):
#     Time_Worth = [float(Time_Worth[i]) for i in range(0,Time_Worth.size)]
# else:
#     Time_Worth = float(Time_Worth)

# if isinstance(Annual_Miles_Avg, np.ndarray):
#     Annual_Miles_Avg = [float(Annual_Miles_Avg[i]) for i in range(0,Annual_Miles_Avg.size)]
# else:
#     Annual_Miles_Avg = float(Annual_Miles_Avg)

# if isinstance(Car_Price, np.ndarray):
#     Car_Price = [float(Car_Price[i]) for i in range(0,Car_Price.size)]
# else:
#     Car_Price = float(Car_Price)

# #Global Inputs, Calculated
# Daily_Miles_Avg = [(Annual_Miles_Avg[i]/365) for i in range(0,len(Annual_Miles_Avg))]   # as miles/day
# Trip_Dist_Avg = [Daily_Miles_Avg[i]/Num_Trips_Avg for i in range(0,len(Daily_Miles_Avg))]
# Trip_Time_Avg = [Daily_Miles_Avg[i]/avgMPH(Annual_Miles_Avg[i])/Num_Trips_Avg*60 for i in range(0,len(Daily_Miles_Avg))] #as minutes

# ######################################################################################################################
# #Traditional Car Ownership Model
# #Expenses to Purchase Car, Hardcoded
# #Car_Price = 24000 #as $
# Sales_Tax_Percent = .0625 #as decimal percent
# Purchase_Fees = 250  # as $ for maybe title transfer
# Down_Payment = 5000 #as $
# Loan_Terms = 48 #as months
# AIR = .03 #as decimal percent

# Sales_Tax_Percent = float(Sales_Tax_Percent)
# Purchase_Fees = float(Purchase_Fees)
# Down_Payment = float(Down_Payment)
# Loan_Terms = float(Loan_Terms)
# AIR = float(AIR)
# #Expenses to Purchase Car, Calculated
# #Sales_Tax_Num = Car_Price*Sales_Tax_Percent #as $
# #Loan_Amount = Car_Price - Down_Payment #as $
# #used in car_ownership now
# MIR = AIR/12
# #Monthly_PMT = (MIR*Loan_Amount*(1+MIR)**Loan_Terms)/(((1+MIR)**Loan_Terms)-1)
# #included in Car_Ownership Model

# ##################################################################################################################
# #Direct Expenses to Own Car Avg'd over 10 years
# Maint_Repairs_Exp = 1250 # as $
# Insurance_Exp = 1000 # as $
# Registr_Taxes_Exp = 150 # as $
# Parking_Exp = 200 # as $
# Maint_Repairs_Exp = float(Maint_Repairs_Exp)
# Insurance_Exp = float(Insurance_Exp)
# Registr_Taxes_Exp = float(Registr_Taxes_Exp)
# Parking_Exp = float(Parking_Exp)

# #Now in Car_Ownership
# #Fuel_Exp = [Annual_Miles_Avg[i]/MPG[i]*Fuel_Price[i] for i in range(0,simsize-1)]
# #Total_DE = Maint_Repairs_Exp+Insurance_Exp+Registr_Taxes_Exp+Parking_Exp+Fuel_Exp

# #####################################################################################################################
# #-----------Uber Model
# #-----Expenses for Hiring
# Fare_Base = 2 # as $
# Fare_Per_Mile = 1.50 # as $/mile
# Fare_Per_Minute = .20 # as $/minute
# # Fare_Base = 1 # as $
# # Fare_Per_Mile = .750 # as $/mile
# # Fare_Per_Minute = .10 # as $/minute
# #In Uber_Model

# #In Uber_Model
# #Uber_Exp_Annual = Uber_Exp_Daily*365
# Min_Fee_Uber = 6.0 # as $
# Cancel_Fee_Uber = 8.0 # as $
# Fare_Base = float(Fare_Base)
# Fare_Per_Mile = float(Fare_Per_Mile)
# Fare_Per_Minute = float(Fare_Per_Minute)

# #Uber_Trip_Avg = Fare_Base+Fare_Per_Mile*Trip_Dist_Avg+Fare_Per_Minute*Trip_Time_Avg
# #Uber_Trip_Avg = float(Uber_Trip_Avg)
# #Uber_Exp_Daily =  Uber_Trip_Avg*Num_Trips_Avg


# #Uber_Exp_Annual = float(Uber_Exp_Annual)
# Min_Fee_Uber = float(Min_Fee_Uber)
# Cancel_Fee_Uber = float(Cancel_Fee_Uber)

# #---Annual Value of Time
# Wait_Time_Uber_Avg = 5.0 # as minutes
# Wait_Time_Uber_Avg = float(Wait_Time_Uber_Avg)
# #Add to Uber Model
# #Val_Time_Daily_Uber = (Wait_Time_Uber_Avg*Num_Trips_Avg)/60*Time_Worth
# #Val_Time_Annual_Uber = Val_Time_Daily_Uber*365
# ###############################################################################################################
# #Annual Indirect Expenses Avg'd over 10 years
# Property_Tax_Garage_IDE= 300 # as $ Garage might be worth $15k; if property tax 2%; garage $300/year
# Garage_Repair_IDE_Ann = 200 # as $
# Property_Tax_Garage_IDE = float(Property_Tax_Garage_IDE)
# Garage_Repair_IDE_
