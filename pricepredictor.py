a = [1,2,4,5,90,44]
n = len(a)
for i in range(0,n):
    if a[i] > a[i+1]:
        a[i],a[i] = a[i+1],a[i+1]
        print(a)

    