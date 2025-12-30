from funciones import run_model

fileNameParam='datos/Spei_CodiVSTipoDeCambio_2019_2025.csv'
columnsParam=[1,2,3]
trainSizeParam=0.95

print('lstm --------------------------------------------------------------------')
algoritmoParam='lstm'
run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam)

print('lm --------------------------------------------------------------------')
algoritmoParam='lm'
run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam)

print('lasso --------------------------------------------------------------------')
algoritmoParam='lasso'
run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam)

print('decisionTree --------------------------------------------------------------------')
algoritmoParam='decisionTree'
run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam)

print('svr --------------------------------------------------------------------')
algoritmoParam='svr'
run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam)

print('rf --------------------------------------------------------------------')
algoritmoParam='rf'
run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam)
print('final --------------------------------------------------------------------')
