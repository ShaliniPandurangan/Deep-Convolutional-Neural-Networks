function res = Conve(img)
map = hot(64);
[l,w] = size(map);
a = img;
a = double(a);
a(a==0) = 1;
ci = ceil(l*a/max(a(:))); 
[il,iw] = size(a);
r = zeros(il,iw); 
g = zeros(il,iw);
b = zeros(il,iw);
r(:) = map(ci,1);
g(:) = map(ci,2);
b(:) = map(ci,3);
res = zeros(il,iw,3);
res(:,:,1) = r; 
res(:,:,2) = g; 
res(:,:,3) = b;
