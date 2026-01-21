from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=0,
    # separator=''
)

text = """
Run the following queries over on the sample classicmodels database, understand them and explain what each query does. 
1. SELECT A productCode, A.productName, B.orderNumber FROM products AINNER JOIN orderDetails B on A.productCode = B .productCode;
2. SELECT c customerNumber, customerName,orderNumber, o.statusFROM customers cLEFT JOIN orders o ON c.customerNumber = o.customerNumber;
3. SELECT o customerNumber, orderNumber, o.status, customerName FROM orders oRIGHT JOIN customers c ON o.customerNumber = c.customerNumber; 
4. SELECT c.customerNumber, c.customerName, c.salesRepEmployeeNumber, e.lastName,e.firstNameFROM customers c LEFT OUTER JOIN employees e ON c.salesRepEmployeeNumber = e.employeeNumber;

"""

#use splitter.split_documents() for splitting loaded documents objects
result = splitter.split_text(text)

# print(type(result))
for res in result:
    print(res)