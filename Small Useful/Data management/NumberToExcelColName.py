columnName = ''

dividend = 1827
while (dividend > 0):

    modulo = (dividend) % 26
    dividend = (dividend - modulo) / 26
    columnName = chr(int(modulo)+96) + columnName



print(columnName)