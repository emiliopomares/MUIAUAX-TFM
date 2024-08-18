import torch
import torch.nn as nn

class VoxelNet_v4(nn.Module):
    """
    VoxelNet, U-Net inspired network which will
    map a 6 channel, stereo RGB image into a
    3d 64x64x64 occupation probability map
    """
    def __init__(self, in_channels, out_channels, steps=5):
        super(VoxelNet_v4, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        self.steps = steps
        self.relu = nn.ReLU()
        self.mode = "occupancy"

        features = [2 ** (i+4) for i in range(steps)]

        # Encoder
        for feature in features:
            #print(f" addinf out feature {feature}")
            self.encoder.append(
                    DoubleConv2D(in_channels, feature),
            )
            in_channels = feature

        out_ch = in_channels
        #print("We start with out_channels: ", out_channels)
        
        # Decoder
        for i in range(1,6):
            # Let's try the last layer trick
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose3d(out_ch*2, out_ch, kernel_size=2 if i<5 else 3, stride=2 if i<5 else 3),
                    DoubleConv3D(out_ch, out_ch//2)
                )
            )
            out_ch = out_ch//2
        
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.final_conv_1 = nn.Conv3d(64, 32, kernel_size=1)
        self.final_conv_2 = nn.Conv3d(32, 16, kernel_size=1)
        self.final_conv_3 = nn.Conv3d(16, 4, kernel_size=1)
        self.final_conv_4 = nn.Conv3d(4, out_channels, kernel_size=1)

        self.pose_dense_1 = nn.Linear(2 ** (self.steps+4-1), 256)
        self.pose_dense_2 = nn.Linear(256, 64)
        self.pose_dense_3 = nn.Linear(64, 16)
        self.pose_dense_4 = nn.Linear(16, 7)

    def set_mode(mode):
        freeze_pose = True
        if mode=='occupancy':
            freeze_pose = True
        elif mode=='pose':
            freeze_pose = False
        else:
            raise ValueException("Unknown mode")
        # Freeze layer1 and layer2
        for param in self.encoder.parameters():
            param.requires_grad = freeze_pose
        for param in self.decoder.parameters():
            param.requires_grad = freeze_pose
        for param in self.residual_connections.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_1.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_2.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_3.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_4.parameters():
            param.requires_grad = freeze_pose
        for param in self.pose_dense_1.parameters():
            param.requires_grad = !freeze_pose
        for param in self.pose_dense_2.parameters():
            param.requires_grad = !freeze_pose
        for param in self.pose_dense_3.parameters():
            param.requires_grad = !freeze_pose
        for param in self.pose_dense_4.parameters():
            param.requires_grad = !freeze_pose

    def forward(self, x):
        
        encoder_outputs = []

        # Encoder
        for module in self.encoder:
            x = module(x)
            x = self.pool(x)
            encoder_outputs.append(x)
        
        if self.mode == "pose":
            x_pose = x[:,:,0,0]
            x_pose = self.pose_dense_1(x_pose)
            x = self.relu(x_pose)
            x_pose = self.pose_dense_2(x_pose)
            x = self.relu(x_pose)
            x_pose = self.pose_dense_3(x_pose)
            x = self.relu(x_pose)
            x_pose = self.pose_dense_4(x_pose)
            return x_pose
        
        x = x.unsqueeze(-1)
        
        # Decoder
        for i in range(1,6):
            residual_connection = encoder_outputs[-i]
            inflated_connection = copy_inflate(residual_connection)
            x = torch.cat([x, inflated_connection], dim=1)
            x = self.decoder[i-1](x)
        
        x = self.final_conv_1(x)
        x = self.relu(x)
        x = self.final_conv_2(x)
        x = self.relu(x)
        x = self.final_conv_3(x)
        x = self.relu(x)
        x = self.final_conv_4(x).squeeze(dim=1)

        return x



