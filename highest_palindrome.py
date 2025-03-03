def highestValuePalindrome(s, n, k):
    # Write your code here
    L = []
    L.extend([int(i) for i in s])
    print(s)
    print(L)
    while k>=1:
        for i in range(n//2):
            if k>1:
                if L[i]==9 and L[n-1-i]!=9:
                    L[n-1-i] = 9
                    k-=1
                elif L[i]!=9 and L[n-1-i]==9:
                    L[n-1-i] = 9
                    k-=1
                elif L[i]!=9 and L[n-1-i]!=9:
                    L[i] = L[n-1-i] = 9
                    k-=2        
            else:
                if L[i]> L[n-1-i]:
                    L[n-1-i] = L[i]
                    k -= 1
                    return L 
                elif L[i]< L[n-1-i]:
                    L[i] = L[n-1-i]
                    k -= 1        
                    return L 
    return L                  
    
s = '12345'
a = highestValuePalindrome(s,5,3)

print(str(a))