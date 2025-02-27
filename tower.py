def tower(start, target, extra, n):
    if n == 1:
        print("Move 1 from", start, "to", target)
        return
    else:
        tower(start, extra, target, n-1)
        print("Move", n, "from", start, "to", target)
        tower(extra, target, start, n-1)    

tower("A", "B", "C", 5)        
