-- @author Sachin Mehta

-- Utility to convert the pascal context data to the required format

require 'image'

function convertNameToIndex(classes)
  local classMod = {}
  for k, v in pairs(classes) do
    if classMod[v] == nil then
      classMod[v] = k
    end
  end
  return classMod
end

function excludedList(classes, exclude)
  exclude = convertNameToIndex(exclude)
  for k, v in pairs(classes) do
     local classToIgnore = exclude[k]
     if classToIgnore ~= nil then
        torch.remove(classes, k)
     end
     
  end
  --print(classes)
  os.exit()
end

function convertPCon()
  local classes = { 'Void', 'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car',  'cat', 'chair', 'cow',  'diningtable',  'dog', 'horse',
                    'motorbike',  'person',  'pottedplant', 'sheep',  'sofa',  'train',  'tvmonitor','bag',
                    'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'clothes', 'computer', 'cup',
                    'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light',
                    'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate', 'road', 'rock', 'shelves',
                    'sidewalk', 'sky', 'snow', 'bedcloth', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'}
                    
  
                    
  local classesToExclude = {'aeroplane', 'bird', 'boat', 'bottle', 'chair', 'dinningtable', 'sofa', 'tvmonitor', 'bag',
                            'bed', 'book', 'cabinet', 'ceiling', 'clothes', 'computer', 'cup', 'door', 'floor', 'flower', 'food',
                            'keyboard', 'light', 'mouse', 'curtain', 'plate', 'shelves', 'snow', 'bedcloth', 'window', 'wood'
                            }
   
  local newClassList =  { 'Void', 'bicycle', 'bus', 'car',  'cat', 'cow',  'dog', 'horse',
                    'motorbike',  'person',  'pottedplant', 'sheep',  'train',
                    'bench', 'building', 'fence', 'grass', 'ground', 'mountain', 'platform', 'sign', 'road', 'rock', 
                    'sidewalk', 'sky', 'track', 'tree', 'truck', 'wall', 'water'}
                 
  classes = convertNameToIndex(classes)  
  --newClassList = convertNameToIndex(newClassList)              
                                        
  local imExtn = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
  
  local dir = './trainannot/'
  
  local findOptions = ' -iname "*.' .. imExtn[1] .. '"'
  for i=2,#imExtn do
     findOptions = findOptions .. ' -o -iname "*.' .. imExtn[i] .. '"'
  end
  
  
  local f = io.popen('find -L ' .. dir .. findOptions)
  
  local i = 0
  local new_map = {}
  while true do
      local line = f:read('*line')
      if not line then break end

      --read label image
      label = image.load(line, 1, 'byte')
      label:add(1)
      for k, v in pairs(classesToExclude) do
        local classToIgnore = classes[v]
        if classToIgnore ~= nil then
          label[label:eq(classToIgnore)] = 1 -- assign background
        end
      end
      local label_new = image.load(line, 1, 'byte')
      label_new:fill(1)
      for k,v in pairs(newClassList) do
        local index = classes[v]
        if index ~= nil then
          label_new[label:eq(classes[v])] = k
          
        end
       
      end
      image.save(line, label_new:byte())
   end
   
   
               

end

convertPCon()