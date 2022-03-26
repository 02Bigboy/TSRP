%%%%%%%%%%%%%%%%%%%%%%%%%
% 这个程序用来挑伪标签中置信度不高的样本
% 返回值n_covey：置信度不高的样本位置
function [n_covey] = unshuffle_label_2(Zt, Cls, r_wj)
          C = length(unique(Cls));               
          n_covey = [];
          a1=0;
          a2=0;
          a3=0;
          Cls_new = Cls;
          for c = reshape(unique(Cls),1,C)
              ny=[];
              fprintf('%d',c)
              X = Zt(Cls==c,:);                     
              n = 1:sum(Cls==c);                
              if length(n)<=3
                  continue;
              end
              sim = zeros(length(n),length(n));
              for i=1:(length(n)-1)
                  for j=(i+1):length(n)
                      fz = X(i,:)*X(j,:)';
                      fm = sqrt(X(i,:)*X(i,:)')*sqrt(X(j,:)*X(j,:)');
                      sim(i,j) = fz/fm;
                  end
              end
              abab = sort(setdiff(sim,0));
              nab = floor(r_wj*length(abab));
              if nab<=0
                  continue;
              end
              thr = abab(nab);              
              sim = sim>=thr;                       
              pic = zeros(1,length(n));          
              for i=1:length(n)
                  pic(i) = sum(sim(i,:))+sum(sim(:,i));
              end
              wz = find(pic==max(pic));            
 
              if length(wz)>=1
                  ny = [ny,wz(1)];
                  wz = wz(1);
                  H = find(sim(wz,:)==1);
                  L = find(sim(:,wz)==1);
                  if length(H)>0
                      ny = [ny,H];  
                  end
                  sim(wz,:) = 0;
                  if length(L)>0
                      ny = [ny,L'];    
                  end
                  sim(:,wz) = 0;
                  while sum(sum(sim)) && (length(H)>0||length(L)>0)
                      h = [];
                      l = [];
                      pre = [H';L];
                      if length(H)>0
                          for i = 1:length(H)
                              ii = H(i);  % ii = setdiff(H(i),pre);  %
                              HH = find(sim(ii,:)==1);
                              HH = setdiff(HH,pre);
                              if length(HH)>0
                                  ny = [ny,HH];
                                  h = [h,HH];
                              end
                              sim(ii,:) = 0;
                              LL = find(sim(:,ii)==1);
                              LL = setdiff(LL,pre);
                              if length(LL)>0
                                  ny = [ny,LL'];
                                  
                                  l = [l;LL];
                              end 
                              sim(:,ii) = 0;
                              pre_later = [h';l];
                              pre = [pre;pre_later];
                          end
                      end
                      if length(L)>0
                          for i = 1:length(L)
                              ii = L(i); % ii = setdiff(L(i),pre);  % ii = L(i);
                              LL = find(sim(:,ii)==1);
                              LL = setdiff(LL,pre);
                              if length(LL)>0
                                  ny = [ny,LL'];
                                  l = [l;LL];
                              end
                              sim(:,ii) = 0;
                              HH = find(sim(ii,:)==1);
                              HH = setdiff(HH,pre);
                              if length(HH)>0
                                  ny = [ny,HH];
                                  
                                  h = [h,HH];
                              end 
                              sim(ii,:) = 0;
                              pre_later = [h';l];
                              pre = [pre;pre_later];
                          end
                      end
                      H = h;
                      L = l;
                  end                                 % 到这里应该就全部选完了,最终Xpic域Ypic就是我们选出的目标域较好的伪标签数据
              end
              nn_location =find(Cls==c);
              no = setdiff(n,ny);
              nn_cover = nn_location(no);
              n_covey = [n_covey;nn_cover];
              a1=a1+length(n);
              a2=a2+length(no);
              a3=a3+length(ny);
          end
 end
