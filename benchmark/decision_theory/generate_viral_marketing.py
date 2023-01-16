import random

base = '''
buy(spaghetti,a) ; buy(steak,a) :- shops(a), target(a).
buy(spaghetti,b) ; buy(beans,b) :- shops(b), target(b).

utility(target(a),-2). utility(target(b),-2).
utility(buy(spaghetti,a),6). utility(buy(steak,a),-11).
utility(buy(spaghetti,b),-3). utility(buy(beans,b),7).
'''

n_products = 100

for size in range(4, 6):
    fp = open(f"viral_{size}.lp", "w")
    
    for i in range(0, size):
        fp.write(f"0.3::shops_a{i}.\n")

    fp.write('\n')

    for i in range(0, size):
        fp.write(f"decision target_a{i}.\n")
    
    fp.write('\n')
    

    products_a = [random.randint(0, n_products - 1) for _ in range(size)]
    products_b = [random.randint(0, n_products - 1) for _ in range(size)]

    i = 0
    for pair in zip(products_a, products_b):
        fp.write(f"buy(p({pair[0]}),a{i}) ; buy(p({pair[1]}),a{i}) :- shops_a{i}, target_a{i}.\n")
        i += 1
    
    fp.write("\n")
    
    for el in range(size):
        fp.write(f"utility(target_a{el}, {random.randint(-5,5)}).\n")

    fp.write("\n")

    # products = list(set(products_b + products_a))
    i = 0
    for pair in zip(products_a, products_b):
        fp.write(f"utility(buy(p({pair[0]}),a{i}),{random.randint(-10,10)}).\n")
        fp.write(f"utility(buy(p({pair[1]}),a{i}),{random.randint(-10,10)}).\n")
        i += 1
        fp.write('\n')
    
    fp.close()
# print(products)