import torch
 
def orthogonal_reg(lamda,Ld1,Ld2,Hd1,Hd2):
    m0 = [0,0,0,0]
    m2 = [0,0]
    m1 = torch.tensor(m0)
    m3 = torch.tensor(m2)
    m1 = m1.cuda()
    m3 = m3.cuda()
    Ld1 = Ld1.squeeze(0).squeeze(0)
    Ld1 = Ld1.cuda()
    Ld2 = Ld2.squeeze(0).squeeze(0)
    Ld2 = Ld2.cuda()
    Hd1 = Hd1.squeeze(0).squeeze(0)
    Hd1 = Hd1.cuda()
    Hd2 = Hd2.squeeze(0).squeeze(0)
    Hd2 = Hd2.cuda()
    K11 = torch.cat([Ld1[0,:],m1,Ld1[1,:],m3],dim=0)
    K12 = torch.cat([m3,Ld1[0,:],m1,Ld1[1,:]],dim=0)
    K21 = torch.cat([Ld2[0,:],m1,Ld2[1,:],m3],dim=0)
    K22 = torch.cat([m3,Ld2[0,:],m1,Ld2[1,:]],dim=0)
    K31 = torch.cat([Hd1[0,:],m1,Hd1[1,:],m3],dim=0)
    K32 = torch.cat([m3,Hd1[0,:],m1,Hd1[1,:]],dim=0)
    K41 = torch.cat([Hd2[0,:],m1,Hd2[1,:],m3],dim=0)
    K42 = torch.cat([m3,Hd2[0,:],m1,Hd2[1,:]],dim=0)
    K = torch.stack([K11,K12,K21,K22,K31,K32,K41,K42],dim=1).t()
    K = K.cuda()
    #####print(K.size())
    A = torch.mm(K,K.t())-torch.eye(8).cuda()
    norm_F = torch.linalg.norm(A, ord='fro')
    R = lamda * norm_F
    return R
    