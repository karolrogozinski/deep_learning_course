# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.


"""
This code is simplified and modified to our pourposes version of official N-BEATS implementation
"""
import torch
import torch.nn as nn


class TrendBasis:
    def __init__(self, degree, input_size, forecast_size):
        self.coef_size = degree + 1
        t_back = torch.linspace(-1, 1, input_size)
        t_for = torch.linspace(-1, 1, forecast_size)

        self.P_back = torch.vstack([t_back**d for d in range(self.coef_size)])
        self.P_for = torch.vstack([t_for**d  for d in range(self.coef_size)])

    def backcast(self, theta): return theta @ self.P_back
    def forecast(self, theta): return theta @ self.P_for


class SeasonalityBasis:
    def __init__(self, harmonics, input_size, forecast_size):
        self.coef_size = 2 * harmonics
        t_back = torch.linspace(0, 2 * torch.pi, input_size)
        t_for = torch.linspace(0, 2 * torch.pi, forecast_size)

        self.F_back = torch.vstack(
            [torch.sin((h + 1) * t_back) for h in range(harmonics)] +
            [torch.cos((h + 1) * t_back) for h in range(harmonics)]
        )
        self.F_for = torch.vstack(
            [torch.sin((h + 1) * t_for) for h in range(harmonics)] +
            [torch.cos((h + 1) * t_for) for h in range(harmonics)]
        )

    def backcast(self, theta): return theta @ self.F_back
    def forecast(self, theta): return theta @ self.F_for


class NBeatsBlock(nn.Module):
    def __init__(self, hidden_size, input_size, basis):
        super().__init__()
        self.basis = basis
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2 * basis.coef_size)
        )

    def forward(self, x):
        theta = self.fc(x)
        theta_b, theta_f = theta.chunk(2, dim=1)
        backcast = self.basis.backcast(theta_b)
        forecast = self.basis.forecast(theta_f)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, input_size, forecast_size=1, hidden_size=256,
                 harmonics=20, poly_degree=2):
        super().__init__()
        self.season_block = NBeatsBlock(hidden_size, input_size,
                                        basis=SeasonalityBasis(harmonics, input_size, forecast_size))
        self.trend_block = NBeatsBlock(hidden_size, input_size,
                                       basis=TrendBasis(poly_degree, input_size, forecast_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        b1, f1 = self.season_block(x)
        res = x - b1
        _, f2 = self.trend_block(res)
        return f1 + f2
