-- @author Sachin Mehta


--- ----
-- Module to parse command line arguments
-- @module opts



local opts = {}

--- --------------------------------------------------
-- A function to parse the command line arguments
-- @function [parent=#opts] parse
-- @param #string arg command line arguments
-- --------------------------------------------------

function opts.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Multi-Scale Residual Network')
  cmd:text()
  cmd:text('Command-Line options:')
  --general options
  cmd:option('-seed', 0, 'random seed')
  --training related options
  cmd:option('-lr', 0.01, 'Initial Learning Rate')
  cmd:option('-d', 10, 'learning rate decay factor')
  cmd:option('-de', 30, 'decay learning rate by x epochs')
  cmd:option('-w', 5e-4, 'weight decay')
  cmd:option('-optimizer', 'adam', 'adam or sgd')
  cmd:option('-m', 0.9, 'momentum')
  cmd:option('-batchSize', 1, 'Batch Size')
  cmd:option('-resume', 'false' , 'resume training')
  cmd:option('-resumeEpoch', 1, 'epoch from where to resume training')
  cmd:option('-optimize', 1, 'optimize the model')
  cmd:option('-maxEpoch', 100, 'Maximum number of epochs')
  --network related options
  cmd:option('-o', 1, 'Optimize the network using optnet')
  cmd:option('-disp', 0, 'Print model and generate graph')
  --data related options
  cmd:option('-snap', './results/', 'save the models here')
  cmd:option('-datapath', './data/CamVid', 'Path to the dataset')
  cmd:option('-dataset', 'cv', 'Which dataset (cv for camvid, pas for pascal, pcon for pascalcontext)')
  cmd:option('-dataCache', './cache/', 'Path to directory for caching data related properties')
  cmd:option('-imWidth', 480, 'Image Width (480 for camvid, 224 for pascal, 512 for pascalcontext)')
  cmd:option('-imHeight', 384, 'Image Height (384 for camvid, 224 for pascal, 512 for pascalcontext)')
  cmd:option('-aug', 'false' , 'Data augmentation')
  cmd:option('-augType', 2, 'Which augmentation? (2 for h-flip, 3 - h-flip + translate, 4 - h-flip + translate + cropScale, 5 - h-flip + translate + cropScaleFlip, 6 - h-flip + translate + cropScaleFlip + v-flip ')
  --model related options
  cmd:option('-model', 'model/MResNet.lua', 'Model File')
  cmd:option('-modelType', 2, 'type of the model (1 for M-Plain, 2 for M-RiR, and 3 for M-Hyper')
  cmd:option('-encModel', 'model/encoder_mres.t7', 'encoder model trained on imagenet')
  
  
  local opt = cmd:parse(arg or {})
  opt.resume = opt.resume ~= 'false'
  opt.aug = opt.aug ~= 'false'

  --check if snapshot directory exist or not. If not, then create it
  if not paths.dirp(opt.snap) and not paths.mkdir(opt.snap) then
    cmd:error('Error: Unable to create snapshot directory: '.. opt.snap .. '\n')
  end
  --check if data directory exist or not
  if not paths.dirp(opt.datapath) then
    cmd:error('Error: Data directory does not exist : '.. opt.datapath .. '\n')
  end

  return opt
end

return opts