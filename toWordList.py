filename = 'text8_1024_4.hash'
filename3 = 'text8_4096_7.hash'
filename2 = 'test.hash'

file = open(filename, 'r')
file2write = open(filename+'.list', 'w')
wordCount = 0
wordHash = {}

while True:
    oneline = file.readline()
    if not oneline: break
    if oneline == '\n': break
        
    nums = oneline.strip('\n')
    nums = nums.split(',')

    nums = tuple([int(i) for i in nums])
    if not nums in wordHash:
        wordHash[nums] = wordCount
        wordCount = wordCount+1
        file2write.write(oneline)
file.close()
file2write.close()

print('count = '+ str(wordCount))