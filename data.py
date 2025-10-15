import numpy as np
import math
import random

class Store(object):
	deliveryEventCount = 0
	arrivalEventCount = 0

	def __init__(self, numPerHour):
		self.customers = 0
		self.lostCustomers = 0		
		self.arrivalRate = numPerHour / 60.0
		self.maxInventory = 500
		self.deliveryRate = 1.0/((self.maxInventory/self.arrivalRate) * 60.0)
		self.inventory = self.maxInventory
		self.waitTime = 0.0
		self.outOfStock = 0

	@staticmethod
	def getArrivalEventCount():
		return Store.arrivalEventCount
	
	@staticmethod
	def getDeliveryEventCount():
		return Store.deliveryEventCount

	def getCustomerCount(self):
		return self.customers
	
	def getWaitTime(self):
		return self.waitTime

	def getStockOutCount(self):
		return self.outOfStock

	def getInventory(self):
		return self.inventory

	def getLostCustomers(self):
		return self.lostCustomers

	def getNextArrival(self):
		return -math.log(1.0 - random.random()) / (self.arrivalRate)

	def getNextDelivery(self): 
		return -math.log(1.0 - random.random()) / (self.deliveryRate)

	def getTravelTime(self):
		return np.random.normal(10, 5)
		
	def getMaxInventory(self):
		return self.maxInventory

	def getNeededInventory(self):
		return self.maxInventory - self.inventory

	@staticmethod
	def increaseDeliveryEventCount():
		Store.deliveryEventCount += 1

	@staticmethod
	def increaseArrivalEventCount():
		Store.arrivalEventCount += 1

	def increaseStockOut(self):
		self.outOfStock += 1
	
	def increaseInventory(self):
		self.inventory += (self.maxInventory - self.inventory) 
		Store.deliveryEventCount -= 1

	def increaseWaitTime(self, wait):
		self.waitTime += wait

	def decreaseInventory(self):
		self.inventory -= 1
		Store.arrivalEventCount -= 1
	
	def increaseCustomers(self):
		self.customers += 1

	def increaseLostCustomers(self):
		self.lostCustomers += 1

	def serviceTime(self):
		return np.random.normal(2, 1)
		
class Supplier(object):
	def __init__(self):
		self.inventory = 10000
	
	def increaseInventory(self):
			self.inventory += (10000 - self.inventory)

	def fulfillTime(self):
		#one day with a variance 8 hours 
		#pair with Store's increaseInventory
		return np.random.normal(1440, 480) 
		
	def decreaseInventory(self):
		self.inventory -= 500

	def getInventory(self):
		return self.inventory

class Main(object):
	def fulfillTime(self):
		#two days with a variance of 8 hours 
		#paired with Supplier's increaseInventory 
		return np.random.normal(2880, 480)	

