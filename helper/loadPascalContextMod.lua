-- @author Sachin Mehta

--- ---
-- File to load Pascal Context Dataset and Cache it
-- @module loadPascalContext

require 'image'

-- load the training and test files
local trainFile = opt.datapath .. '/train.txt'
local valFile = opt.datapath .. '/test.txt'

--local classesName = { 'Void', 'aeroplane', 'bicycle', 'bird', 'boat',
-- 'bottle', 'bus', 'car',  'cat', 'chair', 'cow',  'diningtable',  'dog', 'horse',
-- 'motorbike',  'person',  'pottedplant', 'sheep',  'sofa',  'train',  'tvmonitor','bag',
--'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'clothes', 'computer', 'cup',
--'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light',
--'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate', 'road', 'rock', 'shelves',
--'sidewalk', 'sky', 'snow', 'bedcloth', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'}


local classesName = { 'Void', 'bicycle', 'bus', 'car',  'cat', 'cow',  'dog', 'horse',
                    'motorbike',  'person',  'pottedplant', 'sheep',  'train',
                    'bench', 'building', 'fence', 'grass', 'ground', 'mountain', 'platform', 'sign', 'road', 'rock', 
                    'sidewalk', 'sky', 'track', 'tree', 'truck', 'wall', 'water'}
               
local classes = #classesName

--- ----
-- Function to check if file exists or not
-- @function [parent=#loadPascalContext] check_file 
-- @param #string name File name
-- @return #boolean Boolean indicating whether file exists or not
-- 
local function check_file(name)
   local f=io.open(name,"r")
   if f~=nil then 
    io.close(f) 
    return true 
   else 
    return false 
   end
end

trainImFileList = {}
trainLblFileList = {}
--for modified dataset, lableAddVal = 0
labelAddVal = 0

-- compute the mean and standard deviation for training data
-- do it offline
local mean = {0.4572846486672, 0.43745873801202, 0.40406985094371}
local std = {0.26742718836252, 0.26436273357389, 0.27838534104728}

--parse the training data
if not check_file(trainFile) then
  print('Training file does not exist: ' .. trainFile)
  os.exit()
else
  lineNo = 0
  for line in io.lines(trainFile) do
    local col1, col2 = line:match("([^,]+),([^,]+)")
    trainImFileList[lineNo] =opt.datapath .. col1
    trainLblFileList[lineNo] = opt.datapath .. col2
    
    local rgbIm = image.load(trainImFileList[lineNo])
    --scale the rgb image
    rgbIm = image.scale(rgbIm, opt.imWidth, opt.imHeight)
    
    local labelIm = image.load(trainLblFileList[lineNo], 1, 'byte')
    --scale the label image using simple interpolation
    labelIm = image.scale(labelIm, opt.imWidth, opt.imHeight, 'simple')
    labelIm:add(labelAddVal)
    assert(torch.max(labelIm) <= classes and torch.min(labelIm) > 0, 'Label values should be between 1 and number of classes: max ' .. torch.max(labelIm) .. ' min: ' .. torch.min(labelIm))
    
    lineNo = lineNo + 1
  end
  assert(table.getn(trainImFileList) == table.getn(trainLblFileList), 'Number of images and labels are not equal')
end


--parse the validation data
valImFileList = {}
valLblFileList = {}

if not check_file(valFile) then
  print('Validation file does not exist: ' .. valFile)
  os.exit()
else
  lineNo = 0
  for line in io.lines(valFile) do
    local col1, col2 = line:match("([^,]+),([^,]+)")
    valImFileList[lineNo] =opt.datapath .. col1
    valLblFileList[lineNo] = opt.datapath .. col2
    
    local rgbIm = image.load(valImFileList[lineNo])
    --scale the rgb image
    rgbIm = image.scale(rgbIm, opt.imWidth, opt.imHeight)
    
    local labelIm = image.load(valLblFileList[lineNo], 1, 'byte')
    --scale the label image using simple interpolation
    labelIm = image.scale(labelIm, opt.imWidth, opt.imHeight, 'simple')
    labelIm:add(labelAddVal)
    assert(torch.max(labelIm) <= classes and torch.min(labelIm) > 0, 'Label values should be between 1 and number of classes: max ' .. torch.max(labelIm) .. ' min: ' .. torch.min(labelIm))
    
    lineNo = lineNo + 1
  end
  assert(table.getn(valImFileList) == table.getn(valLblFileList), 'Number of images and labels are not equal')
end

--cache the training and validation data information
dataCache = {}
dataCache.trainIm = trainImFileList
dataCache.trainlbl = trainLblFileList
dataCache.valIm = valImFileList
dataCache.vallbl = valLblFileList
dataCache.mean = mean
dataCache.std = std
dataCache.classes = classes
dataCache.labelAddVal = labelAddVal

if not paths.dirp(opt.cacheDir) and not paths.mkdir(opt.cacheDir) then
  cmd:error('Error: Unable to create a cache directory: '.. opt.cacheDir .. '\n')
end

--save the details about the dataset
torch.save(opt.dataCacheFileName, dataCache)