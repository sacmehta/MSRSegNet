-- @author Sachin Mehta

local nn = require 'nn'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true



----------------------------------------------------------------------
print '==> define parameters'
--~ 
local classes = opt.classes

-- shortcuts for layers -------------------
local Convolution = cudnn.SpatialConvolution --cudnn
local DeConvolution = cudnn.SpatialFullConvolution--cudnn
local Avg = cudnn.SpatialAveragePooling--cudnn
local ReLU = nn.RReLU --cudnn.ReLU --cudnn
local Max = cudnn.SpatialMaxPooling
local UnPool = cudnn.SpatialMaxUnpooling
local SBatchNorm = nn.SpatialBatchNormalization
local Dropout = nn.SpatialDropout
local Dilated = nn.SpatialDilatedConvolution

local iChannels


--- -----
-- Function to create shortcurt connection 
-- @function [parent=#MResNet] shortcutConnectionDec
function shortcutConnectionDec(nInputPlane, nOutputPlane, stride)
     -- 1x1 de-convolution
     return nn.Sequential()
        :add(DeConvolution(nInputPlane, nOutputPlane, 1, 1,stride,stride,0,0, stride - 1, stride -1))
        :add(SBatchNorm(nOutputPlane))
end

--- ------
-- Function to create multi-scale residual block
-- @function [parent=#MResNet] multiScaleResidualBlockDec

function multiScaleResidualBlockDec(n, stride)
  local nInputPlane = iChannels
  local mul = 2
  iChannels = n / mul
  depth_layer = n/4

  local s = nn.Sequential()
  s:add(DeConvolution(nInputPlane,depth_layer,1,1,stride,stride,0,0, stride - 1, stride -1))
  s:add(SBatchNorm(depth_layer))
  s:add(ReLU(1/8, 1/3, true))
  s:add(Convolution(depth_layer,depth_layer,3,3,1,1,1,1))
  s:add(SBatchNorm(depth_layer))
  
  local s1 = nn.Sequential()
  s1:add(DeConvolution(nInputPlane,depth_layer,1,1,stride,stride,0,0, stride - 1, stride -1))
  s1:add(SBatchNorm(depth_layer))
  s1:add(ReLU(1/8, 1/3, true))
  s1:add(Convolution(depth_layer,depth_layer,5,5,1,1,2,2))
  s1:add(SBatchNorm(depth_layer))
  
  local concatDepth = nn.DepthConcat(2)
  concatDepth:add(s)
  concatDepth:add(s1)

  return nn.Sequential()
     :add(nn.ConcatTable()
        :add(concatDepth)
        :add(shortcutConnectionDec(nInputPlane, iChannels, stride)))
     :add(nn.CAddTable(true))
     :add(ReLU(1/8, 1/3, true))
end

--- ------
-- Function to load the pretrained encoder (imagenet)
-- @function [parent=#MResNet] encoder

function encoder()
  enc = torch.load(opt.encModel)
  --drop the fully connected layers
  enc.modules[20] = nil
  enc.modules[19] = nil
  enc.modules[18] = nil
  return enc
end

--- --------
---- Function to create the decoder
-- @function [parent=#MResNet] decoder

function decoder()
  dropout_ratio = 0.2
  dec = nn.Sequential()
  iChannels = 2048
  dec:add(multiScaleResidualBlockDec(2048, 2))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(1024, 2))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(512, 1))
  dec:add(Dropout(dropout_ratio))
  dec:add(multiScaleResidualBlockDec(256, 2))
  dec:add(Dropout(dropout_ratio))
  dec:add(Convolution(128,64,3,3, 1,1,1,1))
  dec:add(SBatchNorm(64))
  dec:add(ReLU(1/8, 1/3, true))
  dec:add(Dropout(dropout_ratio))
  dec:add(DeConvolution(64,32,1,1,2,2,0,0, 1, 1))
  dec:add(Convolution(32, 32, 3, 3, 1, 1, 1, 1))
  dec:add(SBatchNorm(32))
  dec:add(ReLU(1/8, 1/3, true))
  dec:add(Dropout(dropout_ratio))
  
  local function ConvInit(name)
    for k,v in pairs(dec:findModules(name)) do
    local n = v.kW*v.kH*v.nOutputPlane
    v.weight:normal(0,math.sqrt(2/n))
    if cudnn.version >= 4000 then
      v.bias = nil
      v.gradBias = nil
     else
      v.bias:zero()
     end
    end
  end
  local function BNInit(name)
    for k,v in pairs(dec:findModules(name)) do
     v.weight:fill(1)
     v.bias:zero()
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  ConvInit('cudnn.SpatialFullConvolution')
  ConvInit('nn.SpatialFullConvolution')
  ConvInit('nn.SpatialDilatedConvolution')
  BNInit('cudnn.SpatialBatchNormalization')
  BNInit('nn.SpatialBatchNormalization')
  return dec
end

modelTrain = opt.modelType
local model
if modelTrain== 1 then
  print('Finetuning M-Plain')
  model = nn.Sequential()
  model:add(encoder())
  model:add(decoder())
  
  model:add(DeConvolution(32,classes,1,1,2,2,0,0, 1, 1))
  model:add(Convolution(classes,classes,3,3,1,1,1,1))
elseif modelTrain == 2 then
  print('Finetuning M-RiR')
  model_enc = encoder()
  model_dec = decoder()
  
  model = nn.Sequential()
  model:add(model_enc.modules[1])
  model:add(model_enc.modules[2])
  model:add(
    nn.Sequential()
    :add(
      nn.ConcatTable()
      :add(
        nn.Sequential()
          :add(model_enc.modules[3])
          :add(model_enc.modules[4])
          :add(model_enc.modules[5])
          :add(nn.Sequential()
          :add(
            nn.ConcatTable()
            :add(
              nn.Sequential()
              :add(model_enc.modules[6])
              :add(model_enc.modules[7])
              :add(model_enc.modules[8])
              :add(model_enc.modules[9])
              :add(
                nn.Sequential()
                  :add(
                    nn.ConcatTable()
                    :add(
                      nn.Sequential()
                      :add(model_enc.modules[10])
                      :add(model_enc.modules[11])
                      :add(
                        nn.Sequential()
                        :add(
                          nn.ConcatTable()
                          :add(
                            nn.Sequential()
                            :add(model_enc.modules[12])
                            :add(model_enc.modules[13])
                            :add(
                              nn.Sequential()
                              :add(
                                nn.ConcatTable()
                                :add(
                                  nn.Sequential()
                                  :add(
                                    nn.Sequential()
                                    :add(model_enc.modules[14])
                                    :add(model_enc.modules[15])
                                    :add(
                                      nn.ConcatTable()
                                      :add(
                                        nn.Sequential()
                                        :add(model_enc.modules[16])
                                        :add(model_enc.modules[17])
                                        :add(model_dec.modules[1])
                                      )
                                      :add(
                                        nn.Sequential()
                                        --:add(Convolution(1024, 1024, 1, 1, 1, 1))
                                        :add(nn.Identity())
                                      )
                                    )
                                    :add(nn.CAddTable(true))
                                    :add(ReLU(1/8, 1/3, true))
                                    :add(Dropout(0.2))
                                  )
                                  :add(model_dec.modules[3])
                                )
                                --:add(Convolution(512, 512, 1, 1, 1, 1))
                                :add(nn.Identity())
                              )
                              :add(nn.CAddTable(true))
                              :add(ReLU(1/8, 1/3, true))
                              :add(Dropout(0.2))
                            )       
                            :add(model_dec.modules[5])
                          )
                          --:add(Convolution(256, 256, 1, 1, 1, 1))
                          :add(nn.Identity())
                        )
                        :add(nn.CAddTable(true))
                        :add(ReLU(1/8, 1/3, true))
                        :add(Dropout(0.2))
                      )
                      :add(model_dec.modules[7])
                      
                    )
                    --:add(Convolution(128, 128, 1, 1, 1, 1))
                    :add(nn.Identity())
                  )
                  :add(nn.CAddTable(true))
                  :add(ReLU(1/8, 1/3, true))
                  :add(Dropout(0.2))
                )
                :add(model_dec.modules[9])
                :add(model_dec.modules[10])
              )
            --:add(Convolution(64, 64, 1, 1, 1, 1))
            :add(nn.Identity())
          )
          :add(nn.CAddTable(true))
          :add(ReLU(1/8, 1/3, true))
          
        )
        :add(model_dec.modules[13])
          :add(model_dec.modules[14])
        :add(model_dec.modules[15])
      )
      --:add(Convolution(32, 32, 1, 1, 1, 1))
      :add(nn.Identity())
    )
    :add(nn.CAddTable(true))
    :add(model_dec.modules[16])
    :add(Dropout(0.2))
  )
  model:add(DeConvolution(32,classes,1,1,2,2,0,0, 1, 1))
  model:add(Convolution(classes,classes,3,3,1,1,1,1))
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
    if cudnn.version >= 4000 then
      v.bias = nil
      v.gradBias = nil
     else
      v.bias:zero()
     end
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  ConvInit('nn.SpatialFullConvolution')
  ConvInit('cudnn.SpatialFullConvolution')
elseif modelTrain == 3 then
  print('Finetuning M-Hyper')
  model_enc = encoder()
  model_dec = decoder()
  
  model = nn.Sequential()
  model:add(model_enc.modules[1])
  model:add(model_enc.modules[2])
  model:add(model_enc.modules[3])
  model:add(model_enc.modules[4])
  model:add(model_enc.modules[5])
  model:add(model_enc.modules[6])
  model:add(model_enc.modules[7])
  model:add(
    nn.Sequential()
    :add(
        nn.ConcatTable()
        :add(
          nn.Sequential()
          :add(model_enc.modules[8])
          :add(model_enc.modules[9])
          :add(
            nn.ConcatTable()
            :add(
                nn.Sequential()
                :add(model_enc.modules[10])
                :add(model_enc.modules[11])
                :add(
                    nn.ConcatTable()
                    :add(
                      nn.Sequential()
                      :add(model_enc.modules[12])
                      :add(model_enc.modules[13])
                      :add(
                        nn.ConcatTable()
                        :add(
                          nn.Sequential()
                          :add(model_enc.modules[14])
                          :add(model_enc.modules[15])
                          :add(model_enc.modules[16])
                          :add(model_enc.modules[17])
                          :add(model_dec.modules[1])
                          :add(model_dec.modules[2])
                        :add(model_dec.modules[3])
                        :add(model_dec.modules[4])
                        :add(
                            nn.ConcatTable()
                            :add(
                              nn.Sequential()
                              :add(model_dec.modules[5])
                              :add(model_dec.modules[6])
                              :add(
                                nn.ConcatTable()
                                :add(
                                  nn.Sequential()
                                  :add(model_dec.modules[7])
                                  :add(model_dec.modules[8])
                                  :add(
                                    nn.ConcatTable()
                                    :add(
                                      nn.Sequential()
                                      :add(model_dec.modules[9])
                                      :add(model_dec.modules[10])
                                      :add(model_dec.modules[11])
                                      :add(model_dec.modules[12])
                                      :add(
                                        nn.ConcatTable()
                                        :add(
                                            nn.Sequential()
                                            :add(model_dec.modules[13])
                                            :add(model_dec.modules[14])
                                            :add(model_dec.modules[15])
                                            :add(model_dec.modules[16])
                                            :add(model_dec.modules[17])
                                            :add(DeConvolution(32,classes,1,1,2,2,0,0, 1, 1))
                                            :add(Convolution(classes,classes,3,3,1,1,1,1))
                                        :add(SBatchNorm(classes))
                                          )
                                        :add(
                                            nn.Sequential()
                                            :add(DeConvolution(64, classes, 1, 1, 2, 2, 0,0,1,1))
                                            :add(DeConvolution(classes,classes, 1, 1, 2, 2, 0,0,1,1))
                                        :add(SBatchNorm(classes))

                                          )
                                        )
                                      :add(nn.CAddTable(true))
                                    )
                                    :add(
                                      nn.Sequential()
                                      :add(DeConvolution(128,classes,1,1,2,2,0,0,1,1))
                                      :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                                  :add(SBatchNorm(classes))
                                    )
                                  )
                                  :add(nn.CAddTable(true))
                                )
                                :add(
                                  nn.Sequential()
                                  :add(DeConvolution(256,classes,1,1,2,2,0,0,1,1))
                                  :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                                  :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                              :add(SBatchNorm(classes))
                                )
                              )
                              :add(nn.CAddTable(true))
                            )
                            :add(
                              nn.Sequential()
                              :add(DeConvolution(512, classes, 1, 1, 2, 2, 0,0,1,1))
                              :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                              :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                          :add(SBatchNorm(classes))
                            )
                          )
                          :add(nn.CAddTable(true))
                        )
                        :add(
                          nn.Sequential()
                          :add(DeConvolution(512, classes, 1, 1, 2, 2, 0,0,1,1))
                          :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                          :add(DeConvolution(classes, classes, 1, 1, 2, 2, 0,0,1,1))
                      :add(SBatchNorm(classes))
                        )
                      )
                      :add(nn.CAddTable(true))
                    )
                    :add(
                      nn.Sequential()
                      :add(DeConvolution(256, classes, 1, 1, 2, 2, 0,0,1,1))
                      :add(DeConvolution(classes,classes,1,1,2,2,0,0,1,1))
                      :add(DeConvolution(classes,classes,1,1,2,2,0,0,1,1))
                  :add(SBatchNorm(classes))
                    )
                  )
                  :add(nn.CAddTable(true))
                )
            :add(
              nn.Sequential()
              :add(DeConvolution(128, classes, 1, 1, 2, 2, 0,0,1,1))
              :add(DeConvolution(classes,classes,1,1,2,2,0,0,1,1))
            :add(SBatchNorm(classes))
            )
          )
          :add(nn.CAddTable(true))
        )
      :add(
        nn.Sequential()
        :add(DeConvolution(64, classes, 1, 1, 2, 2, 0,0,1,1))
        :add(DeConvolution(classes,classes,1,1,2,2,0,0,1,1))
      :add(SBatchNorm(classes))
      )
    )
    :add(nn.CAddTable(true))
  )
  model:add(ReLU(1/8, 1/3, true))
  model:add(Convolution(classes,classes,3,3,1,1,1,1))
  
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
    if cudnn.version >= 4000 then
      v.bias = nil
      v.gradBias = nil
     else
      v.bias:zero()
     end
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  ConvInit('nn.SpatialFullConvolution')
  ConvInit('cudnn.SpatialFullConvolution')
else
  print('No model selected')
  print('Exiting')
  os.exit()
end

model:cuda()

return model
