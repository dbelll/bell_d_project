def loop(start, end, factor, string):
    reps = (end-start)/factor
    code = ""
    if reps > 0:
        code += "for(int __i__={0}; __i__<{1}; __i__+= {2})".format(start, start + factor*reps - 1, factor)
        code += "{\n"
        for j in range(factor):
            code += "   "+string.format("(__i__+{0})".format(j))
        code += "\n}\n"
    for k in range(start + factor*reps, end):
        code += string.format(k)
    return code