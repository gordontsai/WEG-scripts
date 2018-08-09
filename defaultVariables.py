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
