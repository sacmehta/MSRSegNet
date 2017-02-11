-- @author Sachin Mehta

require 'optim'
require 'xlua'
require 'image'

confusionMatTest = optim.ConfusionMatrix(confClasses)

--save the training error to files
local valLogger = optim.Logger(paths.concat(opt.snap, 'error_test_' .. opt.resumeEpoch .. '.log'))

local inputs = torch.Tensor(1, 3, opt.imHeight, opt.imWidth)
local targets = torch.Tensor(1, opt.imHeight, opt.imWidth)

inputs = inputs:cuda()
targets = targets:cuda()

--- ----
-- Function to test the network
-- @function [parent=#test] test
-- @param #number epoch Epoch number
-- @param #table dataset Table containing the information about the dataset

local function test(epoch, dataset)
  local time = sys.clock()
  model:evaluate()
  
  valSize = table.getn(dataset.valIm)
  validationErr = 0
  for i = 1,valSize do
    xlua.progress(i, valSize)
    local rgbImg = image.load(dataset.valIm[i]):float()
    rgbImg = image.scale(rgbImg, opt.imWidth, opt.imHeight)
    
    rgbImg[1]:add(-dataset.mean[1])
    rgbImg[2]:add(-dataset.mean[2])
    rgbImg[3]:add(-dataset.mean[3])

    rgbImg[1]:div(dataset.std[1])
    rgbImg[2]:div(dataset.std[2])
    rgbImg[3]:div(dataset.std[3])
    
    local lblImg = image.load(dataset.vallbl[i], 1, 'byte'):float()
    lblImg = image.scale(lblImg, opt.imWidth, opt.imHeight,  'simple')
    lblImg:add(dataset.labelAddVal)
    
    inputs[1] = rgbImg
    targets[1] = lblImg
    
    local output = model:forward(inputs)
    local err = criterion:forward(output,targets)
    validationErr = validationErr + err
    
    local _, pred = output:max(2)
    confusionMatTest:batchAdd(pred:view(-1), targets:view(-1))
  end
  
  time = (sys.clock() - time)/valSize
  validationErr = validationErr / valSize
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')  
  print('Validation Error: ' .. validationErr)
    
  valLogger:add{['Validation Error '] = validationErr, 
                ['Epoch'] = epoch} 
  
  
  --save the model
  local filename = paths.concat(opt.snap, 'model-' .. epoch .. '.t7')
  print('saving model file: ' .. filename)
  if opt.dpt = true then
  	torch.save(filename, model:get(1):clearState()) --remove the dpt layer
  else
	torch.save(filename, model:clearState())
  end
  --save the confusion matrix
  local filenameCon = paths.concat(opt.snap, 'con-' .. epoch .. '.txt')
  print('saving confusion matrix: ' .. filenameCon)
  local fileCon = io.open(filenameCon, 'w')
  
  fileCon:write("--------------------------------------------------------------------------------\n")
  fileCon:write("Training:\n")
  fileCon:write("================================================================================\n")
  fileCon:write(tostring(confusionMatTrain))
  fileCon:write("\n--------------------------------------------------------------------------------\n")
  fileCon:write("Testing:\n")
  fileCon:write("================================================================================\n")
  fileCon:write(tostring(confusionMatTest))
  fileCon:write("\n--------------------------------------------------------------------------------")
  fileCon:close()
  
  print('\n')
  confusionMatTest:zero()
  confusionMatTrain:zero()
  collectgarbage()
end

return test
