def UTSP(similarity: torch.Tensor,
         threshold: int = 0.8):
  '''
  思想：在类内中，利用相似性和生成树，选出置信度高的样本
  '''
    len_t = len(similar)
    position = torch.arange(0, len_t, dtype=torch.int64)   # 位置信息
    similar = similar - torch.diag_embed(torch.diag(similar))     # 主对角元素取0
    # 只保留相似度大于最大的threshold的部分
    threshold = threshold
    max_similarity = similar.max().max() * threshold
    similar[similar < max_similarity] = 0
    similar[similar >= max_similarity] = 1
    dgree = sum(similar)
    index_i = dgree.argmax()
    span_tree = []
    span_tree.append(index_i.item())
    next_layer = position[similar[index_i, :] > torch.zeros(similar[index_i, :].size(), device=similar.device)].numpy().tolist()
    similar[index_i, :] = 0
    similar[:, index_i] = 0
    # tic4 = time.time()
    # print('L {}'.format(tic4-tic3))
    # # TODO 搜索层数
    num_layer = 2
    layer = 0
    while sum(sum(similar)) and (len(next_layer) > 0):
        span_tree.extend(next_layer)
        layer += 1
        if layer >= num_layer:
            break
        similar[:, next_layer] = 0
        cur_layer = position[similar[next_layer, :].sum(0) > torch.zeros(similar[next_layer[0], :].size(), device=similar.device)].numpy().tolist()
        similar[next_layer, :] = 0
        # similar[:, next_layer] = 0  # 这个放前面取感觉可以可以不用下面的diff
        # next_layer = np.setdiff1d(cur_layer, next_layer)
        next_layer = cur_layer
    return span_tree
