#more complex model



class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual


class ResNetV2(nn.Module):
    BLOCK_UNITS = {'r50': [3, 4, 6, 3]}

    def __init__(self, block_units, width_factor=1, num_classes=100):
        super().__init__()
        wf = width_factor

        # Modified root for CIFAR-100 (32x32)
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64 * wf, kernel_size=3, stride=1, padding=1, bias=False)),
            ('gn', nn.GroupNorm(32, 64 * wf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', self._make_block(64 * wf, 256 * wf, block_units[0], stride=1)),
            ('block2', self._make_block(256 * wf, 512 * wf, block_units[1], stride=2)),
            ('block3', self._make_block(512 * wf, 1024 * wf, block_units[2], stride=2)),
            ('block4', self._make_block(1024 * wf, 2048 * wf, block_units[3], stride=2)),
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048 * wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(2048 * wf, num_classes)),
        ]))

    def _make_block(self, cin, cout, units, stride):
        layers = [PreActBottleneck(cin, cout, stride=stride)]
        for _ in range(1, units):
            layers.append(PreActBottleneck(cout, cout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        x = self.head(x)
        return x