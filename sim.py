import sys
import data
import random
from tabulate import tabulate

try:
	rates = sys.argv[1].split(',')
	rates = map(float, rates)
	factor = int(sys.argv[2])
	timeUnit = sys.argv[3]
	
except (IndexError, ValueError) as e:
	print("Invalid number or type of arguments")
	sys.exit(1)

time = {'hour': 60.0 , 'day': 1440.0, 'week': 10080.0, 'month': 43829.1 , 'year': 525949.0}
runTime = factor * time[timeUnit]
visitNext = False
currentTime = 0.0
eventQueue = dict( )
system = list( )
stores = [ ]

for rate in rates:
	stores.append(data.Store(rate))

suppliers = [data.Supplier(), data.Supplier()]
main = data.Main()

while currentTime < runTime:
	for store in stores:		
		if visitNext and store.getInventory() > 0:
			service = store.serviceTime()
			eventQueue[currentTime + service ] = store.decreaseInventory
			store.increaseCustomers()
			store.increaseWaitTime(service)
			data.Store.increaseArrivalEventCount()
			visitNext = False

		if visitNext and store.getInventory() == 0:
			supplier = suppliers[int(round(random.uniform(0,1)))]
			mfulfill = main.fulfillTime()
			sfulfill = supplier.fulfillTime()
			service = store.serviceTime()

			if supplier.getInventory() == 0: 
				eventQueue[currentTime + mfulfill] = supplier.increaseInventory
				eventQueue[currentTime + mfulfill] = supplier.decreaseInventory
				eventQueue[currentTime + mfulfill + sfulfill] = store.increaseInventory
				eventQueue[currentTime + mfulfill + sfulfill + service] = store.decreaseInventory
				store.increaseCustomers()
				store.increaseStockOut()
				store.increaseWaitTime(mfulfill + sfulfill + service)
				data.Store.increaseArrivalEventCount()
				data.Store.increaseDeliveryEventCount()
				visitNext = False

			else:
				eventQueue[currentTime] = supplier.decreaseInventory
				eventQueue[currentTime + sfulfill] = store.increaseInventory
				eventQueue[currentTime + sfulfill + service] = store.decreaseInventory
				store.increaseCustomers()
				store.increaseStockOut()
				store.increaseWaitTime(sfulfill + service)
				data.Store.increaseArrivalEventCount()
				data.Store.increaseDeliveryEventCount()
				visitNext = False

		if not visitNext and store.getInventory() > 0:
			service = store.serviceTime()
			eventQueue[currentTime + store.getNextArrival() + service] = store.decreaseInventory
			store.increaseCustomers()
			store.increaseWaitTime(service)
			data.Store.increaseArrivalEventCount()

		if not visitNext and store.getInventory() == 0:
			supplier = suppliers[int(round(random.uniform(0,1)))]
			mfulfill = main.fulfillTime()
			sfulfill = supplier.fulfillTime()
			service = store.serviceTime()
			travel = store.getTravelTime()

			if supplier.getInventory() == 0: 
				eventQueue[currentTime + mfulfill] = supplier.increaseInventory
				eventQueue[currentTime + mfulfill] = supplier.decreaseInventory
				eventQueue[currentTime + mfulfill + sfulfill] = store.increaseInventory
				store.increaseCustomers()
				store.increaseStockOut()
				currentTime += travel
				store.increaseWaitTime(travel)
				store.increaseLostCustomers()
				data.Store.increaseArrivalEventCount()
				data.Store.increaseDeliveryEventCount()
				visitNext = True
			
			else:
				eventQueue[currentTime] = supplier.decreaseInventory
				eventQueue[currentTime + sfulfill] = store.increaseInventory
				store.increaseCustomers()
				store.increaseLostCustomers()
				store.increaseStockOut()
				currentTime += travel
				store.increaseWaitTime(travel)
				store.increaseLostCustomers()
				data.Store.increaseArrivalEventCount()
				data.Store.increaseDeliveryEventCount()
				visitNext = True

	keys = eventQueue.keys()
	keys = sorted(keys)

	for key in keys:	
		if data.Store.getArrivalEventCount() == 0 and data.Store.getDeliveryEventCount() > 0:
			break

		else:
			currentTime = key
			eventQueue.pop(key)() 	

categories = ["Stores","Customers", "Nil Stock", "Lost", "Total Wait Time(minutes)", "P(Nil)"]
table = [ ]

for i in range(0, len(stores)):
	table.append([i + 1, 
	stores[i].getCustomerCount(), 
	stores[i].getStockOutCount(), 
	stores[i].getLostCustomers(), 
	stores [i].getWaitTime(), 
	float(stores[i].getStockOutCount())/stores[i].getCustomerCount()])

print (tabulate(table, headers=categories, tablefmt="grid", numalign="right"), "\n")

