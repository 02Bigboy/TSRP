% 将选出的置信度高的目标域样本与源域结合一起训练一个分类器纠正置信度不高的样本
% r_wj：置信度因子
% TT：寻找置信度的次数
function Yt = remedy_plab_3(Zs,Ys,Zt,Cls,Yg,r_wj,TT)
         

         Cl = Cls;
         Yt = Cls;

         Ztp = Zt;
         n=TT; %挑选的级数
         n_m = 1:length(Cls);
         Z = Zs;
         Y = Ys;
         for i=1:n
             nn = unshuffle_label_2(Ztp,Cls,r_wj);
             nn_o = n_m(nn);
             n_y = setdiff(n_m,nn_o);
             Zty =Zt(n_y,:);
             Zto = Zt(nn_o,:);
             Yty = Yt(n_y);% 
             Z = [Z;Zty];
             Y = [Y;Yty];
             knn_model_all = fitcknn (Z,Y) ;
             Yt(nn_o)=knn_model_all.predict(Zto);
             acc_yp = sum(Yg(n_y)==Yt(n_y)) /length(Yt(n_y));
             acc_p = sum(Yg(n_y)==Cl(n_y)) /length(Cl(n_y));
             acc_yo = sum(Yg(nn_o)==Yt(nn_o)) /length(Yt(nn_o));
             acc_o = sum(Yg(nn_o)==Cl(nn_o)) /length(Cl(nn_o));
             acc_now = sum(Yg==Yt)/length(Yt);
             fprintf('\n---%0.4f--%0.4f--%0.4f--%0.4f--%0.4f--',acc_yp,acc_p,acc_yo,acc_o,acc_now)
             if length(nn_o)==0
                 break;
             end
             Ztp = Zt(nn_o,:);  % 将不自信的伪标掐在传入算法，继续挑纠正后的自信的样本
             Cls = Yt(nn_o);
             n_m=nn_o;
         end
        
         fprintf('\n----%d==?%d---\n',length(Cls),length(Yt))
  end
