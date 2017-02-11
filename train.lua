-- @author Sachin Mehta

require 'optim'
require 'xlua'
require 'image'
dataAug = require 'dataAugmentation'


--- ---------------
-- @module Module to train the network
-- 

confClasses = {}
for i=1,opt.classes do
  confClasses[i] = i
end

-- confusion matrix for training data
confusionMatTrain = optim.ConfusionMatrix(confClasses)

--save the training error to files
local trainLogger = optim.Logger(paths.concat(opt.snap, 'error_train_' .. opt.resumeEpoch .. '.log'))

-- extract models parameters
parameters, gradParameters = model:getParameters()

--specify the optimizer
if opt.optimizer == 'adam' then
  optimState = {
    learningRate = opt.lr,
    momentum = opt.m,
    learningRateDecay = 1e-7,
    weightDecay = opt.w
  }
  optimMethod = optim.adam
elseif opt.optimizer == 'sgd' then
  optimState = {
   learningRate = opt.lr,
   momentum = opt.m,
   learningRateDecay = 1e-7,
   nesterov = true,
   dampening = 0.0,
   weightDecay = opt.w
  }
  optimMethod = optim.sgd
else
  print('Please add the optimizer in the train file.')
  os.exit()
end

-- tensors to store the batch data
local inputsTr = torch.Tensor(opt.batchSize, 3, opt.imHeight, opt.imWidth)
local targetsTr = torch.Tensor(opt.batchSize, opt.imHeight, opt.imWidth)

inputsTr = inputsTr:cuda()
targetsTr = targetsTr:cuda()

--- ------
-- Fucntion to train the network
-- @function [parent=#train] train
-- @param #number epoch Epoch number
-- @param #table dataset table that contains the details about the dataset such as image/label location

local function train(epoch, dataset)
  
  local time = sys.clock()
  
  --set the model to training state
  model:training()
  
  --compute the size of training data
  trainingSize = table.getn(dataset.trainIm) 
  
  
  -- decay the learning rate after x epochs by d
  if epoch % opt.de == 0 then
    optimState.learningRate = optimState.learningRate/opt.d
  end
  
  print('Training Epoch --> '.. epoch .. ' [LR = ' .. optimState.learningRate .. ']')
  
  -- check if we want to augment the data or not
  local repeatData = 1
  if opt.aug == true then
    -- THis value needs to be changed if we want to need more augmentaion
    repeatData = opt.augType
  end
  
  local epochTrainErr = 0
  
  for rep = 1, repeatData do 
    -- shuffle the data
    shuffle = torch.randperm(trainingSize)
    for t = 1,trainingSize,opt.batchSize do
      xlua.progress(t, trainingSize)
      local idx = 1
      for i = t, math.min(t+opt.batchSize-1, trainingSize) do
        -- load new sample
        local rgbImg = image.load(dataset.trainIm[shuffle[i]]):float()
        rgbImg = image.scale(rgbImg, opt.imWidth, opt.imHeight)
        
        rgbImg[1]:add(-dataset.mean[1])
        rgbImg[2]:add(-dataset.mean[2])
        rgbImg[3]:add(-dataset.mean[3])

        rgbImg[1]:div(dataset.std[1])
        rgbImg[2]:div(dataset.std[2])
        rgbImg[3]:div(dataset.std[3])

        local lblImg = image.load(dataset.trainlbl[shuffle[i]], 1, 'byte'):float()
        lblImg = image.scale(lblImg, opt.imWidth, opt.imHeight,  'simple')
        lblImg:add(dataset.labelAddVal)
        lblImg[lblImg:gt(opt.classes)] = 1


        if rep%2 == 0 then
          -- horizontal flipping
          rgbImg, lblImg = dataAug.flip_h(rgbImg, lblImg)
        elseif rep%3 == 0 then
          --translation
          rgbImg, lblImg = dataAug.translate(rgbImg, lblImg)
        elseif rep%4 == 0 then
          --Cropping + Scaling
          rgbImg, lblImg = dataAug.cropScale(rgbImg, lblImg)
        elseif rep%5 == 0 then
          -- cropping + scaling + flipping
          rgbImg, lblImg = dataAug.cropScaleFlip(rgbImg, lblImg)
        elseif rep%6 == 0 then
          -- vertical flipping
          rgbImg, lblImg = dataAug.flip_v(rgbImg, lblImg)
        end

        inputsTr[idx] = rgbImg
        targetsTr[idx] = lblImg
        idx = idx + 1
      end

      --- ---
      -- Function to do forward and backward computation
      -- @function [parent=#train] feval
      -- @param #tensor parameters Network parameters
      -- @return #float training error for the batch
      -- @return #tensor gradient parameters
      function feval(parameters)
        -- reset gradients
        gradParameters:zero()
        local output = model:forward(inputsTr)
        local err = criterion:forward(output, targetsTr)

        -- estimate df/dW
        local df_do = criterion:backward(output, targetsTr)
        model:backward(inputsTr, df_do)
        
        
        local _, pred = output:max(2)
        confusionMatTrain:batchAdd(pred:view(-1), targetsTr:view(-1))

        epochTrainErr = epochTrainErr + err
        -- return f and df/dX
        return err, gradParameters
      end
      optimMethod(feval, parameters, optimState)
    end
  end

  time = (sys.clock() - time)/trainingSize
  epochTrainErr = (epochTrainErr / (trainingSize/opt.batchSize))/repeatData
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')  
  print('Training Error: ' .. epochTrainErr)
  
  trainLogger:add{['Training Error '] = epochTrainErr, 
                  ['Learning rate'] = optimState.learningRate, 
                  ['Epoch'] = epoch} 
  
  collectgarbage()
  return model, criterion, confusionMatTrain
end

return train
