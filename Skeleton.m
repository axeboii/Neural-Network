global A x1 x2 T1 T2 V1 V2 W1 W2 y1 y2 z1 z2 x0 xhat xhatPleq x02 xhat2 xhatPleq2

%% Determine the matrices to use based on the personal number of the oldest
PersonalNumber = 9804120179;
[A,x1,x2] = getData(PersonalNumber);

T1 = rref(A)

%% Ex 1a
% V1 and V2 should be determined by hand and manually written here

T1 = [1 0 0 2 -5;0 1 0 -4/3 49/3;0 0 1 1/3 -16/3;0 0 0 0 0;0 0 0 0 0;0 0 0 0 0;0 0 0 0 0];
V1 = [2 1 1;3 3 6; 2 3 6;4 4 7;1 2 5;1 1 1;3 2 2];
V2 = [-2 5;4/3 -49/3;-1/3 16/3;1 0;0 1];


%% Ex 1b
% W1 and W2 should be determined by hand and manually written here

T2 = [1 0 0 1 -1 1 2;0 1 0 0 1 -1 -1;0 0 1 1 0 1 1;0 0 0 0 0 0 0;0 0 0 0 0 0 0];
W1 = [2 3 2;1 3 3;1 6 6;3 4 2;1 2 7];
W2 = [-1 1 -1 -2;0 -1 1 1;-1 0 -1 -1;1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];

%% Ex 1c
matris = [W1 V2];
ts = matris\[x1];


y1 = W1*ts(1:3)
z1 = V2*ts(4:5)
x1 = y1 + z1;
zero = y1'*z1

y1 = [5.00 2.35 2.21 7.60 1.64]';
z1 = [4.00 5.65 -2.21 -3.60 -0.64]';

%{
 > What is y1'*z1 ?

'0'

%}

%% Ex 1d

matris = [W2 V1];
ts = matris\[x2];


y2 = W2*ts(1:4)
z2 = V1*ts(5:7)
x2Test = y2 + z2;
zero = y2'*z2

y2 = [3.50 1.00 0.50 -1.00 -0.50 4.00 -3.50]';
z2 = [5.50 7.00 -0.50 5.00 1.50 -2.00 3.50]';

%{
 > What is y2'*z2 ?

"0"

%}

%% Ex 2a,b

Anew = [A eye(7)];
b = ones(7,1);
c = -[ones(1,5) zeros(1,7)]';
basic = [6 7 8 9 10 11 12];

[x,z,u,basic,status,it] = simple(c,Anew,b,basic);


x0 = [0 0 0 0 0 1 1 1 1 1 1 1]';
xhat = [0.25 0 0 0 0 0.50 0.25 0.50 0 0.75 0.75 0.25]';

vec = xhat - x0;
z = Anew*vec;
%{
Describe the formulation of Ps and how simple is called to obtain xhat

Create new A-matrix by adding slack-variables x_6 --> x_12. We do this by
adding an identity matrix to the A-matrix. Transforms Ax<=b to Ax=b.
Create b-vector according to instruction.
Make max-problem to min-problem by multiplying C with -1.


xhat = simple(c,Anew,b,basic)
 
> Show that xhat - x0 is in the nullspace of A ?

"Anew * (xhat - x0) = 0"

%}


%% Ex 2c

constraints = Anew * xhat

xhatPleq = [2.50 0 0 0 0]'
%{
 > Describe which constraints are active ?

"All constraints are active. Contraints = b"

%}


%% Ex 2d

b02 = A(:,5)

[x,z,u,basic,status,it] = simple(c,Anew,b02,basic)

Anew*x

x02 = [1 1 1 1 1 1 1]';
xhat2 = [0 0 0 0 1 0 0 0 0 0 0 0]';
xhatPleq2 =[0 0 0 0 1]';

constraints = Anew*xhat2

%{
 > Describe which constraints are active ?

"All constraints are active"

 > What do we call such solutions?

"A degenarate solution"

%}
