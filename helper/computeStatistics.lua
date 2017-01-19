-- @author Sachin Mehta

--- ---
-- Module to compute mean and std of training dataset
-- @module computeStatistics

require 'image'

imDir = './data/CamVid/train'
fileExtn = 'png'
imWidth = 480
imHeight = 384
local findOptions = ' -iname "*.' .. fileExtn .. '"'

local f = io.popen('find -L ' .. imDir .. findOptions)
local filesList = {}
local lineNo = 0
while true do
  local line = f:read('*line')
  if not line then 
    break 
  end
  table.insert(filesList, line)
end

data = torch.FloatTensor(table.getn(filesList), 3, imHeight, imWidth)

for i = 1, table.getn(filesList) do
  local rawImg = image.load(filesList[i])
  rawImg = image.scale(rawImg, imWidth, imHeight)
  data[i] = rawImg
end

mean = {data[{{}, 1, {}, {}}]:mean(), data[{{}, 2, {}, {}}]:mean(), data[{{}, 3, {}, {}}]:mean()}
std = {data[{{}, 1, {}, {}}]:std(), data[{{}, 2, {}, {}}]:std(), data[{{}, 3, {}, {}}]:std()}
print(mean)
print(std)