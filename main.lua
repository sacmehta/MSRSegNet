-- @author Sachin Mehta


--- -----
-- Module to start training
-- @module main

require 'torch'
require 'paths'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

--- parse the command line arguments
local opts = require 'opts'
opt = opts.parse(arg) --global variable (can be accessed across files)

opt.cacheDir = opt.dataCache .. '/' .. opt.dataset
opt.dataCacheFileName = opt.cacheDir .. '/data.t7'

if opt.dataset == 'cv' then
  if not paths.filep(opt.dataCacheFileName) then
    print('Loading Camvid dataset from loadCamVid')
    require 'loadCamVid'
  else
    print('loading cached Camvid file')
  end
  dataset = torch.load(opt.dataCacheFileName)
elseif opt.dataset == 'pcon' then
  if not paths.filep(opt.dataCacheFileName) then
    print('Loading Pascal Context dataset from loadPascalContext')
    require 'loadPascalContext'
  else
    print('loading cached file')
  end
  dataset = torch.load(opt.dataCacheFileName)
elseif opt.dataset == 'pas' then
  if not paths.filep(opt.dataCacheFileName) then
    print('Loading Pascal dataset from loadPascal')
    require 'loadPascal'
  else
    print('loading cached file')
  end
  dataset = torch.load(opt.dataCacheFileName)
else
  print('Wrong dataset specified. Please check')
  print('Exiting')
  os.exit()
end

--number of classes in the dataset
opt.classes = dataset.classes
print('Dataset has ' .. opt.classes .. ' classes including background')

-- load the model
local models = require 'init'
model, criterion, epochNo = models.modelSetup()

local train = require 'train'
local test = require 'test'

-- start training and validation
for ep = epochNo, opt.maxEpoch do
  model, criterion, confusionMat = train(ep, dataset)
  test(ep, dataset)
  collectgarbage()
end
