def DetectDuplicate(thelist):
  seen = set()
  duplicates =[]
  for x in thelist:
    if x in seen: 
    	duplicates.append(x)
    	# return duplicates

    seen.add(x)
