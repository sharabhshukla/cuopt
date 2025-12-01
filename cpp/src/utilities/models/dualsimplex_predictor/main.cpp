/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include "header.h"

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif
#define N_TARGET    1
#define MAX_N_CLASS 1

const unsigned char is_categorical[] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};
static const int32_t num_class[] = {
  1,
};

int32_t dualsimplex_predictor::get_num_target(void) { return N_TARGET; }
void dualsimplex_predictor::get_num_class(int32_t* out)
{
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t dualsimplex_predictor::get_num_feature(void) { return 18; }
const char* dualsimplex_predictor::get_threshold_type(void) { return "float32"; }
const char* dualsimplex_predictor::get_leaf_output_type(void) { return "float32"; }

void dualsimplex_predictor::predict(union Entry* data, int pred_margin, double* result)
{
  // Quantize data
  for (int i = 0; i < 18; ++i) {
    if (data[i].missing != -1 && !is_categorical[i]) {
      data[i].qvalue = quantize(data[i].fvalue, i);
    }
  }

  unsigned int tmp;
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 30))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += -0.23892987;
            } else {
              result[0] += -0.41049096;
            }
          } else {
            result[0] += -0.32070833;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 16))) {
              result[0] += -0.3301402;
            } else {
              result[0] += -0.27258733;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 158))) {
              result[0] += -0.2263252;
            } else {
              result[0] += -0.03232357;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 244))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.21953094;
            } else {
              result[0] += -0.2782724;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 252))) {
              result[0] += -0.10610497;
            } else {
              result[0] += -0.18065642;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 238))) {
              result[0] += -0.14425437;
            } else {
              result[0] += -0.027287258;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 4))) {
              result[0] += -0.06345942;
            } else {
              result[0] += -0.026511494;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 84))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
              result[0] += -0.20900369;
            } else {
              result[0] += -0.03617132;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += -0.17355898;
            } else {
              result[0] += -0.094911605;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 174))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
              result[0] += -0.064165816;
            } else {
              result[0] += -0.12049206;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 116))) {
              result[0] += -0.11783655;
            } else {
              result[0] += 0.031279184;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 14))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
              result[0] += -0.16023064;
            } else {
              result[0] += -0.01693966;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 92))) {
              result[0] += -0.03719565;
            } else {
              result[0] += 0.002473806;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 180))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 194))) {
              result[0] += 0.015176967;
            } else {
              result[0] += -0.033581402;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 52))) {
              result[0] += 0.060070585;
            } else {
              result[0] += 0.102768324;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 210))) {
      if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 152))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
              result[0] += -0.018653875;
            } else {
              result[0] += 0.21873346;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 158))) {
              result[0] += 0.020794097;
            } else {
              result[0] += 0.0017005502;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 54))) {
              result[0] += -0.15759692;
            } else {
              result[0] += 0.037903294;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += 0.073502585;
            } else {
              result[0] += 0.1481389;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 72))) {
            result[0] += -0.20302705;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 298))) {
              result[0] += -0.0740808;
            } else {
              result[0] += -0.0013614334;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 36))) {
            result[0] += 0.007832853;
          } else {
            result[0] += 0.05149386;
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 334))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 286))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += 0.038612913;
            } else {
              result[0] += -0.011268199;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 246))) {
              result[0] += 0.12736425;
            } else {
              result[0] += 0.057369012;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 270))) {
              result[0] += 0.09891705;
            } else {
              result[0] += 0.1404978;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 200))) {
              result[0] += -0.09467173;
            } else {
              result[0] += 0.058953542;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 98))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 12))) {
              result[0] += 0.24495018;
            } else {
              result[0] += 0.4303154;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.23759733;
            } else {
              result[0] += 0.31459442;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
              result[0] += 0.18160371;
            } else {
              result[0] += 0.119219065;
            }
          } else {
            result[0] += 0.2541432;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 6))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += -0.13811207;
            } else {
              result[0] += -0.39534476;
            }
          } else {
            result[0] += -0.3041533;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 20))) {
            result[0] += -0.29433212;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 78))) {
              result[0] += -0.19736992;
            } else {
              result[0] += -0.25767314;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
              result[0] += -0.19146764;
            } else {
              result[0] += -0.15695396;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
              result[0] += -0.23099203;
            } else {
              result[0] += -0.117888466;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 16))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
              result[0] += -0.1496231;
            } else {
              result[0] += -0.10690468;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 160))) {
              result[0] += -0.027002404;
            } else {
              result[0] += -0.08140517;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 84))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 70))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
              result[0] += -0.18216138;
            } else {
              result[0] += -0.023672506;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 170))) {
              result[0] += -0.12947556;
            } else {
              result[0] += -0.04930939;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 170))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
              result[0] += -0.10542603;
            } else {
              result[0] += -0.05690325;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 70))) {
              result[0] += -0.11127614;
            } else {
              result[0] += 0.029446835;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 24))) {
              result[0] += -0.07772863;
            } else {
              result[0] += -0.02781359;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 86))) {
              result[0] += -0.13588855;
            } else {
              result[0] += -0.0629831;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 180))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 298))) {
              result[0] += 0.008945032;
            } else {
              result[0] += -0.030025298;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 52))) {
              result[0] += 0.053789187;
            } else {
              result[0] += 0.09317514;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 212))) {
      if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 172))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 150))) {
              result[0] += -0.01571287;
            } else {
              result[0] += 0.01397138;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 22))) {
              result[0] += -0.004348435;
            } else {
              result[0] += 0.28950748;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 62))) {
              result[0] += 0.073582016;
            } else {
              result[0] += 0.15677379;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 84))) {
              result[0] += -0.031289082;
            } else {
              result[0] += 0.06539955;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 298))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 14))) {
              result[0] += -0.18994586;
            } else {
              result[0] += -0.066469155;
            }
          } else {
            result[0] += 0.026545838;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 82))) {
            result[0] += 0.0066967257;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 122))) {
              result[0] += 0.04904222;
            } else {
              result[0] += 0.012652091;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 308))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 80))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 54))) {
              result[0] += 0.036342237;
            } else {
              result[0] += -0.0249609;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
              result[0] += 0.10429803;
            } else {
              result[0] += 0.3026218;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 252))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.19158468;
            } else {
              result[0] += -0.07063005;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 264))) {
              result[0] += -0.019653698;
            } else {
              result[0] += 0.029620511;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 328))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 222))) {
              result[0] += 0.16111112;
            } else {
              result[0] += 0.25347957;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 128))) {
              result[0] += 0.084318936;
            } else {
              result[0] += 0.14822438;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 92))) {
              result[0] += 0.22045271;
            } else {
              result[0] += 0.26758388;
            }
          } else {
            result[0] += 0.46477053;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 134))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 4))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += -0.1654226;
            } else {
              result[0] += -0.36575767;
            }
          } else {
            result[0] += -0.28903162;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 18))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 30))) {
              result[0] += -0.24255578;
            } else {
              result[0] += -0.27643654;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
              result[0] += -0.29263017;
            } else {
              result[0] += -0.20599626;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 18))) {
              result[0] += -0.18999758;
            } else {
              result[0] += -0.15109059;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
              result[0] += -0.11279365;
            } else {
              result[0] += -0.027640833;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
              result[0] += -0.12278713;
            } else {
              result[0] += -0.029810125;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 100))) {
              result[0] += -0.21111389;
            } else {
              result[0] += -0.13208562;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 146))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 78))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 42))) {
              result[0] += -0.11854438;
            } else {
              result[0] += -0.17666526;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 64))) {
              result[0] += -0.012537091;
            } else {
              result[0] += -0.081729345;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 80))) {
              result[0] += -0.10848161;
            } else {
              result[0] += -0.06910487;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
              result[0] += -0.048394956;
            } else {
              result[0] += -0.10724862;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 118))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 18))) {
              result[0] += -0.05678125;
            } else {
              result[0] += -0.15847282;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
              result[0] += -0.02826718;
            } else {
              result[0] += 0.016257456;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 170))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 194))) {
              result[0] += 0.009919471;
            } else {
              result[0] += -0.02828365;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 52))) {
              result[0] += 0.048264306;
            } else {
              result[0] += 0.084261395;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 220))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 204))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 154))) {
              result[0] += -0.015983945;
            } else {
              result[0] += 0.0104940515;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
              result[0] += 0.059158504;
            } else {
              result[0] += -0.013891051;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 166))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.05883024;
            } else {
              result[0] += -0.15169065;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 50))) {
              result[0] += 0.017066706;
            } else {
              result[0] += 0.046345506;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 302))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
              result[0] += 0.114555776;
            } else {
              result[0] += 0.40284094;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
              result[0] += 0.027867997;
            } else {
              result[0] += -0.05320992;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 160))) {
              result[0] += 0.02757929;
            } else {
              result[0] += 0.0005367231;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
              result[0] += 0.0925671;
            } else {
              result[0] += -0.04869043;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 334))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 80))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 54))) {
              result[0] += 0.033666294;
            } else {
              result[0] += -0.019405624;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 138))) {
              result[0] += 0.085760355;
            } else {
              result[0] += 0.11623438;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 136))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 78))) {
              result[0] += -0.078748904;
            } else {
              result[0] += -0.03139619;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
              result[0] += 0.020764524;
            } else {
              result[0] += 0.15907182;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 156))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.19595024;
            } else {
              result[0] += 0.25730672;
            }
          } else {
            result[0] += 0.41746393;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 90))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 20))) {
              result[0] += 0.18315254;
            } else {
              result[0] += 0.3391243;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
              result[0] += 0.1157012;
            } else {
              result[0] += 0.20929395;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 108))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 118))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 40))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 74))) {
              result[0] += -0.14199895;
            } else {
              result[0] += -0.0956778;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 140))) {
              result[0] += -0.010430599;
            } else {
              result[0] += 0.019095603;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
              result[0] += -0.3010808;
            } else {
              result[0] += -0.23643827;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
              result[0] += -0.20573525;
            } else {
              result[0] += -0.02070453;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 204))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 68))) {
              result[0] += -0.120953314;
            } else {
              result[0] += -0.075053215;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.023893813;
            } else {
              result[0] += -0.05592366;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += -0.16090266;
            } else {
              result[0] += -0.22481023;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 86))) {
              result[0] += -0.14591117;
            } else {
              result[0] += -0.081503354;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 124))) {
              result[0] += 0.014306935;
            } else {
              result[0] += -0.15783915;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 156))) {
              result[0] += -0.04105756;
            } else {
              result[0] += -0.014158033;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 142))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 82))) {
              result[0] += -0.120835006;
            } else {
              result[0] += -0.073639594;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 60))) {
              result[0] += -0.19654752;
            } else {
              result[0] += -0.14132342;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 192))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 26))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 76))) {
              result[0] += 0.01790429;
            } else {
              result[0] += -0.00943239;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 238))) {
              result[0] += 0.04468555;
            } else {
              result[0] += 0.011115446;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 62))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 270))) {
              result[0] += 0.07178088;
            } else {
              result[0] += 0.011843091;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
              result[0] += 0.0443505;
            } else {
              result[0] += 0.14387575;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 288))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 254))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 148))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 76))) {
              result[0] += -0.04890412;
            } else {
              result[0] += -0.0113124;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 190))) {
              result[0] += 0.015577379;
            } else {
              result[0] += -0.0047443947;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 300))) {
              result[0] += 0.03677338;
            } else {
              result[0] += 0.08133157;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 58))) {
              result[0] += 0.11096779;
            } else {
              result[0] += 0.15190493;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 186))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 70))) {
              result[0] += 0.0974144;
            } else {
              result[0] += 0.012945512;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 228))) {
              result[0] += -0.016480766;
            } else {
              result[0] += 0.020737674;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 176))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += -0.105966404;
            } else {
              result[0] += -0.01857656;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
              result[0] += -0.033622157;
            } else {
              result[0] += 0.042780038;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 334))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 14))) {
            result[0] += -0.0662734;
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 140))) {
              result[0] += 0.07991419;
            } else {
              result[0] += 0.039648328;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 172))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 120))) {
              result[0] += 0.10550159;
            } else {
              result[0] += 0.03400041;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 84))) {
              result[0] += 0.11472396;
            } else {
              result[0] += 0.1589642;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 96))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 8))) {
              result[0] += 0.14915675;
            } else {
              result[0] += 0.3244665;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.17384307;
            } else {
              result[0] += 0.23245731;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
              result[0] += 0.13902494;
            } else {
              result[0] += 0.07743486;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
              result[0] += 0.19507462;
            } else {
              result[0] += 0.12790385;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 14))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += -0.12810235;
            } else {
              result[0] += -0.30201578;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
              result[0] += -0.25788355;
            } else {
              result[0] += -0.21340947;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
              result[0] += -0.21370807;
            } else {
              result[0] += -0.167573;
            }
          } else {
            result[0] += -0.024952482;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 100))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
              result[0] += -0.171652;
            } else {
              result[0] += -0.12910992;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 110))) {
              result[0] += -0.07442858;
            } else {
              result[0] += -0.12269157;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 12))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 28))) {
              result[0] += -0.11061738;
            } else {
              result[0] += -0.06790245;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 160))) {
              result[0] += -0.017483674;
            } else {
              result[0] += -0.059728812;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 146))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 82))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
              result[0] += -0.13760419;
            } else {
              result[0] += -0.01986711;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
              result[0] += -0.0773085;
            } else {
              result[0] += 0.004780628;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 62))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 144))) {
              result[0] += -0.08369659;
            } else {
              result[0] += -0.0037746632;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += -0.033646163;
            } else {
              result[0] += -0.09390003;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 118))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 64))) {
              result[0] += -0.10188579;
            } else {
              result[0] += -0.022887342;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 84))) {
              result[0] += -0.025158664;
            } else {
              result[0] += 0.011464008;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 170))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 204))) {
              result[0] += 0.009676366;
            } else {
              result[0] += -0.025687149;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 52))) {
              result[0] += 0.03910928;
            } else {
              result[0] += 0.06689557;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 220))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 204))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 80))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 160))) {
              result[0] += 0.039443213;
            } else {
              result[0] += 0.20666161;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 128))) {
              result[0] += -0.01548522;
            } else {
              result[0] += 0.015606319;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 150))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 392))) {
              result[0] += -0.0018438485;
            } else {
              result[0] += -0.03514717;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 90))) {
              result[0] += -0.029947493;
            } else {
              result[0] += -0.056570943;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 302))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
              result[0] += 0.15045454;
            } else {
              result[0] += 0.01461363;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
              result[0] += 0.031331427;
            } else {
              result[0] += 0.2708533;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
              result[0] += -0.044682816;
            } else {
              result[0] += 0.019505082;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 278))) {
              result[0] += 0.10829522;
            } else {
              result[0] += 0.07073908;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 312))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += 0.027972206;
            } else {
              result[0] += 0.005189055;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 44))) {
              result[0] += -0.066226505;
            } else {
              result[0] += 0.0877003;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 148))) {
              result[0] += 0.055681467;
            } else {
              result[0] += -0.06665366;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 280))) {
              result[0] += 0.0809778;
            } else {
              result[0] += 0.1468879;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
              result[0] += 0.15474413;
            } else {
              result[0] += 0.20727856;
            }
          } else {
            result[0] += 0.35305986;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 158))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
              result[0] += 0.07976506;
            } else {
              result[0] += 0.1597683;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
              result[0] += 0.13799594;
            } else {
              result[0] += 0.23070168;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 104))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 84))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 40))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.24169908;
            } else {
              result[0] += -0.107800476;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 80))) {
              result[0] += 0.0015814465;
            } else {
              result[0] += -0.14862476;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 82))) {
              result[0] += -0.2477706;
            } else {
              result[0] += -0.1950383;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
              result[0] += -0.17119522;
            } else {
              result[0] += -0.045236852;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 178))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
              result[0] += 0.025135875;
            } else {
              result[0] += -0.08694006;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.029477531;
            } else {
              result[0] += -0.04847209;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 116))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
              result[0] += 0.011227789;
            } else {
              result[0] += -0.09084964;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 82))) {
              result[0] += -0.13153963;
            } else {
              result[0] += -0.18369597;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 152))) {
              result[0] += -0.061374586;
            } else {
              result[0] += -0.111760534;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 276))) {
              result[0] += -0.03116385;
            } else {
              result[0] += -0.085914634;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 142))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
              result[0] += 0.0005968995;
            } else {
              result[0] += -0.08622521;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 70))) {
              result[0] += -0.17008883;
            } else {
              result[0] += -0.11829443;
            }
          }
        }
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 34))) {
              result[0] += 0.0151279615;
            } else {
              result[0] += 0.037952267;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 58))) {
              result[0] += 0.059345204;
            } else {
              result[0] += 0.11877208;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 206))) {
            result[0] += -0.12507677;
          } else {
            result[0] += -0.025324045;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 290))) {
      if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 4))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 102))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 246))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
              result[0] += 0.044818092;
            } else {
              result[0] += -0.01875449;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 148))) {
              result[0] += -0.11071397;
            } else {
              result[0] += -0.059991468;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
              result[0] += 0.049274206;
            } else {
              result[0] += 0.012975462;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
              result[0] += -0.0189122;
            } else {
              result[0] += 0.060573917;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 148))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
              result[0] += 0.02068209;
            } else {
              result[0] += -0.027593452;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 60))) {
              result[0] += -0.008190083;
            } else {
              result[0] += 0.01192194;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 268))) {
              result[0] += 0.02700766;
            } else {
              result[0] += 0.07031421;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 58))) {
              result[0] += 0.09329746;
            } else {
              result[0] += 0.12738243;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 336))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 128))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
              result[0] += 0.04464999;
            } else {
              result[0] += 0.0775819;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 62))) {
              result[0] += -0.05400176;
            } else {
              result[0] += 0.0332122;
            }
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 60))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
              result[0] += 0.06202929;
            } else {
              result[0] += 0.0011356722;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
              result[0] += 0.093274996;
            } else {
              result[0] += 0.12987015;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 124))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
              result[0] += 0.15837678;
            } else {
              result[0] += 0.3200806;
            }
          } else {
            result[0] += 0.34475276;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
              result[0] += 0.09141225;
            } else {
              result[0] += 0.15360029;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.14422126;
            } else {
              result[0] += 0.18972561;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 106))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 102))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 38))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
              result[0] += -0.22463696;
            } else {
              result[0] += -0.17236648;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
              result[0] += -0.15470378;
            } else {
              result[0] += -0.014106827;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 80))) {
              result[0] += -0.104941204;
            } else {
              result[0] += -0.06797316;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
              result[0] += 0.0034419482;
            } else {
              result[0] += -0.04432465;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 220))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 66))) {
              result[0] += -0.09611558;
            } else {
              result[0] += -0.05750701;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.02356928;
            } else {
              result[0] += -0.045556568;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
            result[0] += -0.12279941;
          } else {
            result[0] += -0.16899776;
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 124))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 130))) {
              result[0] += -0.017295085;
            } else {
              result[0] += 0.031519514;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 156))) {
              result[0] += -0.16593443;
            } else {
              result[0] += -0.110011;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 112))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
              result[0] += -0.010664171;
            } else {
              result[0] += -0.032727893;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 142))) {
              result[0] += -0.07378478;
            } else {
              result[0] += -0.13850205;
            }
          }
        }
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 134))) {
              result[0] += 0.007132952;
            } else {
              result[0] += 0.033364255;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 62))) {
              result[0] += 0.053948194;
            } else {
              result[0] += 0.105476275;
            }
          }
        } else {
          result[0] += -0.07528159;
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 288))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 254))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 154))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
              result[0] += 0.017853899;
            } else {
              result[0] += -0.021608775;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 190))) {
              result[0] += 0.012284887;
            } else {
              result[0] += -0.0040449556;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
              result[0] += 0.027741803;
            } else {
              result[0] += 0.063282035;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 56))) {
              result[0] += 0.08446597;
            } else {
              result[0] += 0.116814755;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 200))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 70))) {
              result[0] += 0.0801467;
            } else {
              result[0] += 0.00028896204;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 206))) {
              result[0] += -0.0006912606;
            } else {
              result[0] += 0.038264047;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 152))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += -0.09579014;
            } else {
              result[0] += -0.011496059;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 124))) {
              result[0] += 0.028059239;
            } else {
              result[0] += -0.029746488;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 332))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 60))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 234))) {
              result[0] += 0.04334253;
            } else {
              result[0] += 0.094181366;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
              result[0] += 0.010113914;
            } else {
              result[0] += 0.12312211;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 12))) {
              result[0] += 0.008063223;
            } else {
              result[0] += 0.05981284;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 44))) {
              result[0] += 0.055709254;
            } else {
              result[0] += 0.102501765;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 126))) {
              result[0] += 0.19393599;
            } else {
              result[0] += 0.31415454;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.12953043;
            } else {
              result[0] += 0.17184684;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 152))) {
              result[0] += 0.105829634;
            } else {
              result[0] += 0.05405576;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 342))) {
              result[0] += 0.11501894;
            } else {
              result[0] += 0.14885037;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 142))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 8))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 32))) {
              result[0] += -0.22360039;
            } else {
              result[0] += -0.16961168;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 62))) {
              result[0] += -0.17449044;
            } else {
              result[0] += -0.13112317;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 2))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
              result[0] += -0.19438948;
            } else {
              result[0] += -0.07163301;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 58))) {
              result[0] += -0.12211982;
            } else {
              result[0] += -0.1598212;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 72))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 104))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 8))) {
              result[0] += -0.1360238;
            } else {
              result[0] += -0.08778509;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 172))) {
              result[0] += -0.058016848;
            } else {
              result[0] += -0.01712372;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.10796758;
            } else {
              result[0] += -0.1462996;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 72))) {
              result[0] += -0.026550526;
            } else {
              result[0] += -0.09057524;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 70))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 60))) {
              result[0] += -0.1045504;
            } else {
              result[0] += -0.06835998;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
              result[0] += -0.16272134;
            } else {
              result[0] += -0.10771642;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 236))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
              result[0] += -0.1004568;
            } else {
              result[0] += -0.02189359;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 12))) {
              result[0] += -0.11555834;
            } else {
              result[0] += -0.064784154;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 110))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 78))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 138))) {
              result[0] += -0.09062075;
            } else {
              result[0] += -0.01616345;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 144))) {
              result[0] += -0.046594307;
            } else {
              result[0] += -0.00848129;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 220))) {
              result[0] += 0.048658125;
            } else {
              result[0] += 0.015779605;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.003939248;
            } else {
              result[0] += -0.031186854;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 222))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 254))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 154))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
              result[0] += 0.0046192957;
            } else {
              result[0] += 0.02478583;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
              result[0] += 0.24694644;
            } else {
              result[0] += -0.104915835;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += -0.0028355815;
            } else {
              result[0] += -0.03066038;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 288))) {
              result[0] += -0.038436286;
            } else {
              result[0] += 0.03750279;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 78))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 186))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 318))) {
              result[0] += -0.012060843;
            } else {
              result[0] += -0.0011414163;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 314))) {
              result[0] += -0.008850611;
            } else {
              result[0] += 0.015718905;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 262))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 196))) {
              result[0] += 0.05882001;
            } else {
              result[0] += 0.036198247;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 184))) {
              result[0] += -0.065134026;
            } else {
              result[0] += -0.01131267;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 312))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 122))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
            result[0] += 0.2215176;
          } else {
            result[0] += -0.10633468;
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 240))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
              result[0] += 0.1440095;
            } else {
              result[0] += 0.051038798;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 254))) {
              result[0] += -0.014680609;
            } else {
              result[0] += 0.014171252;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 348))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 224))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 94))) {
              result[0] += 0.056311846;
            } else {
              result[0] += 0.1058489;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 362))) {
              result[0] += 0.037274715;
            } else {
              result[0] += 0.11873694;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 152))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 34))) {
              result[0] += 0.11126106;
            } else {
              result[0] += 0.14717433;
            }
          } else {
            result[0] += 0.26775053;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 134))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 8))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
              result[0] += -0.20885597;
            } else {
              result[0] += -0.16570954;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += -0.13637789;
            } else {
              result[0] += -0.1756009;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 2))) {
            result[0] += -0.21495152;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
              result[0] += -0.15078692;
            } else {
              result[0] += -0.11473024;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += -0.08377748;
            } else {
              result[0] += -0.020055672;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
              result[0] += -0.12106979;
            } else {
              result[0] += -0.092178635;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 16))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 26))) {
              result[0] += -0.07951072;
            } else {
              result[0] += -0.04535841;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 52))) {
              result[0] += 0.015479415;
            } else {
              result[0] += -0.034244347;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 202))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 76))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
              result[0] += -0.10090798;
            } else {
              result[0] += -0.072621964;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
              result[0] += -0.08769948;
            } else {
              result[0] += -0.020480268;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 52))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 158))) {
              result[0] += -0.06127919;
            } else {
              result[0] += 0.0065026283;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 24))) {
              result[0] += -0.06342202;
            } else {
              result[0] += -0.021188881;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 72))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 100))) {
              result[0] += -0.04954914;
            } else {
              result[0] += 0.00018559814;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 44))) {
              result[0] += 0.011565405;
            } else {
              result[0] += -0.011750548;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 146))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
              result[0] += 0.075431995;
            } else {
              result[0] += -0.005443739;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 62))) {
              result[0] += 0.025297;
            } else {
              result[0] += 0.044371624;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 234))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 206))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 112))) {
              result[0] += -0.012151074;
            } else {
              result[0] += 0.0095630875;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.19431071;
            } else {
              result[0] += 0.039510164;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 150))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += -0.002731852;
            } else {
              result[0] += -0.02748406;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
              result[0] += -0.039663028;
            } else {
              result[0] += -0.002819218;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 294))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
              result[0] += 0.012211016;
            } else {
              result[0] += 0.04506604;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 108))) {
              result[0] += -8.271459e-05;
            } else {
              result[0] += -0.031217247;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 10))) {
              result[0] += -0.027022822;
            } else {
              result[0] += 0.015010217;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += 0.08849927;
            } else {
              result[0] += 0.04585655;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 330))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 56))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
              result[0] += 0.015790751;
            } else {
              result[0] += 0.0709383;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 302))) {
              result[0] += 0.014195338;
            } else {
              result[0] += 0.043147992;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 306))) {
              result[0] += 0.027121753;
            } else {
              result[0] += 0.06687878;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 180))) {
              result[0] += 0.06856555;
            } else {
              result[0] += 0.11084541;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 96))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 12))) {
              result[0] += 0.023235757;
            } else {
              result[0] += 0.12572078;
            }
          } else {
            result[0] += 0.24674617;
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 362))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
              result[0] += 0.08457372;
            } else {
              result[0] += 0.04069808;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 372))) {
              result[0] += 0.11791612;
            } else {
              result[0] += 0.07502523;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 142))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 6))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 28))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += -0.054601975;
            } else {
              result[0] += -0.17477903;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 48))) {
              result[0] += -0.00046666022;
            } else {
              result[0] += -0.13438587;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 2))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 8))) {
              result[0] += -0.07183619;
            } else {
              result[0] += -0.17046754;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 120))) {
              result[0] += -0.10375444;
            } else {
              result[0] += -0.14129147;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += -0.07520628;
            } else {
              result[0] += -0.01688747;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 14))) {
              result[0] += -0.11755419;
            } else {
              result[0] += -0.08725922;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 28))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 2))) {
              result[0] += -0.06106487;
            } else {
              result[0] += -0.09542205;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 202))) {
              result[0] += 0.008344304;
            } else {
              result[0] += -0.03182082;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 76))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 64))) {
              result[0] += -0.061504174;
            } else {
              result[0] += -0.13946581;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
              result[0] += -0.13620088;
            } else {
              result[0] += -0.08506644;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 236))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 78))) {
              result[0] += -0.10902418;
            } else {
              result[0] += -0.016639745;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 244))) {
              result[0] += -0.10518378;
            } else {
              result[0] += -0.05821876;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 116))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 90))) {
              result[0] += -0.06888045;
            } else {
              result[0] += -0.0376487;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 88))) {
              result[0] += -0.012150378;
            } else {
              result[0] += 0.015397436;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 216))) {
              result[0] += 0.039432753;
            } else {
              result[0] += 0.013889983;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.0010314513;
            } else {
              result[0] += -0.02555494;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 232))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 206))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 200))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 80))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 160))) {
              result[0] += 0.030333001;
            } else {
              result[0] += 0.17310323;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
              result[0] += 0.02483304;
            } else {
              result[0] += 0.0021163383;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 224))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += -0.017784828;
            } else {
              result[0] += -0.039406788;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 234))) {
              result[0] += -0.0026142828;
            } else {
              result[0] += 0.03959627;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 284))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
              result[0] += 0.019107591;
            } else {
              result[0] += -0.011726064;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
              result[0] += 0.096682;
            } else {
              result[0] += 0.22430603;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += 0.012120399;
            } else {
              result[0] += -0.03992904;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += 0.06968047;
            } else {
              result[0] += 0.04038763;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 304))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
              result[0] += 0.11028786;
            } else {
              result[0] += 0.040963266;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
              result[0] += 0.19369291;
            } else {
              result[0] += 0.0747213;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 78))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.057252772;
            } else {
              result[0] += -0.14311497;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 232))) {
              result[0] += -0.0102287065;
            } else {
              result[0] += 0.016080577;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 330))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
              result[0] += 0.07653522;
            } else {
              result[0] += 0.13459362;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 130))) {
              result[0] += 0.03455645;
            } else {
              result[0] += 0.06964803;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.09500605;
            } else {
              result[0] += 0.12430712;
            }
          } else {
            result[0] += 0.22509204;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 74))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 20))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 48))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.16930752;
            } else {
              result[0] += -0.1269749;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
              result[0] += -0.06503186;
            } else {
              result[0] += -0.12498716;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
              result[0] += -0.1927099;
            } else {
              result[0] += -0.08201694;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 38))) {
              result[0] += -0.068333164;
            } else {
              result[0] += -0.13142657;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 12))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
            result[0] += 0.042084176;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.14493789;
            } else {
              result[0] += -0.09196159;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 98))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
              result[0] += 0.03108189;
            } else {
              result[0] += -0.067129865;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += 0.0054919203;
            } else {
              result[0] += -0.05392629;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 144))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 98))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 30))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
              result[0] += -0.21116984;
            } else {
              result[0] += -0.0651527;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
              result[0] += 0.116855875;
            } else {
              result[0] += 0.050092816;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 148))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 20))) {
              result[0] += -0.08852083;
            } else {
              result[0] += -0.04572432;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 270))) {
              result[0] += -0.0053127306;
            } else {
              result[0] += 0.024463532;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 32))) {
            result[0] += -0.22384882;
          } else {
            result[0] += -0.093135744;
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 16))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 74))) {
              result[0] += -0.031588506;
            } else {
              result[0] += -0.08342363;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 116))) {
              result[0] += 0.012553929;
            } else {
              result[0] += 0.058796555;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 146))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 22))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 300))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 4))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 46))) {
              result[0] += 0.08138124;
            } else {
              result[0] += 0.13923828;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 202))) {
              result[0] += 0.021075254;
            } else {
              result[0] += 0.064726695;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 98))) {
              result[0] += 0.016024219;
            } else {
              result[0] += -0.014299775;
            }
          } else {
            result[0] += -0.04494404;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 86))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 82))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += -0.03340462;
            } else {
              result[0] += -0.002975365;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 130))) {
              result[0] += -0.13237435;
            } else {
              result[0] += -0.098908305;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
              result[0] += -0.019236477;
            } else {
              result[0] += -0.033017647;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
              result[0] += 0.0251749;
            } else {
              result[0] += 0.0012758941;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 230))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.016069753;
            } else {
              result[0] += -0.010248724;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
              result[0] += -0.028139247;
            } else {
              result[0] += -0.06801506;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 164))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 238))) {
              result[0] += 0.031309973;
            } else {
              result[0] += 0.07697828;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 258))) {
              result[0] += 0.03671503;
            } else {
              result[0] += 0.0007923256;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 282))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 86))) {
              result[0] += 0.02349461;
            } else {
              result[0] += -0.00087367493;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
              result[0] += 0.03977532;
            } else {
              result[0] += 0.096183844;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 290))) {
              result[0] += -0.08533253;
            } else {
              result[0] += -0.06052809;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 238))) {
              result[0] += -0.055827565;
            } else {
              result[0] += -0.0045159357;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 152))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 32))) {
              result[0] += -0.17435847;
            } else {
              result[0] += -0.09712316;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 88))) {
              result[0] += -0.121227026;
            } else {
              result[0] += -0.09049445;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 90))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 148))) {
              result[0] += -0.07094587;
            } else {
              result[0] += -0.13034694;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += -0.0074528246;
            } else {
              result[0] += -0.1171114;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 16))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
              result[0] += -0.03242543;
            } else {
              result[0] += -0.09674471;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 136))) {
              result[0] += -0.060467966;
            } else {
              result[0] += -0.08785518;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 28))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 54))) {
              result[0] += -0.047304705;
            } else {
              result[0] += -0.070193194;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 238))) {
              result[0] += -0.027161488;
            } else {
              result[0] += 0.0049651144;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 124))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
              result[0] += -0.05672009;
            } else {
              result[0] += -0.133577;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 240))) {
              result[0] += -0.034619935;
            } else {
              result[0] += -0.074495845;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 130))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
              result[0] += -0.016871883;
            } else {
              result[0] += 0.0060241153;
            }
          } else {
            result[0] += -0.07992915;
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 118))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 256))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 90))) {
              result[0] += -0.059571445;
            } else {
              result[0] += -0.026049837;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 164))) {
              result[0] += 0.0035159283;
            } else {
              result[0] += -0.047432255;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
              result[0] += -0.009207371;
            } else {
              result[0] += 0.026254697;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 28))) {
              result[0] += -0.083354376;
            } else {
              result[0] += -0.025358835;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 234))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 158))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 246))) {
              result[0] += 0.004427155;
            } else {
              result[0] += 0.030195776;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += 0.043274168;
            } else {
              result[0] += 0.0072428077;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 92))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 164))) {
              result[0] += -0.052554864;
            } else {
              result[0] += 0.0033617232;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
              result[0] += -0.037242856;
            } else {
              result[0] += -0.008167746;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 112))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.094510235;
            } else {
              result[0] += 0.01284566;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 26))) {
              result[0] += 0.07989215;
            } else {
              result[0] += 0.17329922;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 76))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += 0.058576096;
            } else {
              result[0] += -0.055972952;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
              result[0] += 0.21819083;
            } else {
              result[0] += 0.09395635;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 130))) {
              result[0] += 0.01516776;
            } else {
              result[0] += 0.033623938;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 106))) {
              result[0] += 0.043214235;
            } else {
              result[0] += 0.119155124;
            }
          }
        } else {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 38))) {
            result[0] += 0.09584941;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
              result[0] += 0.080481224;
            } else {
              result[0] += 0.03998689;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 318))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 126))) {
              result[0] += 0.026821358;
            } else {
              result[0] += -0.031004164;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 202))) {
              result[0] += 0.06501947;
            } else {
              result[0] += 0.19631971;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 180))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 166))) {
              result[0] += 0.07131386;
            } else {
              result[0] += 0.028573189;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 314))) {
              result[0] += 0.083791435;
            } else {
              result[0] += 0.17529456;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 124))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 22))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
              result[0] += -0.01719068;
            } else {
              result[0] += -0.064043045;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
              result[0] += -0.1791604;
            } else {
              result[0] += -0.13283774;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
              result[0] += -0.056413215;
            } else {
              result[0] += -0.14871566;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
              result[0] += -0.09970745;
            } else {
              result[0] += -0.010530283;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 34))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 72))) {
              result[0] += -0.019443454;
            } else {
              result[0] += -0.09627802;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
              result[0] += -0.072033755;
            } else {
              result[0] += -0.029162338;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 56))) {
              result[0] += -0.04072467;
            } else {
              result[0] += -0.07095223;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 80))) {
              result[0] += 9.782781e-05;
            } else {
              result[0] += -0.028554544;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 70))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += -0.10857904;
            } else {
              result[0] += -0.17578779;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 148))) {
              result[0] += -0.05417463;
            } else {
              result[0] += -0.08070096;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 300))) {
              result[0] += -0.06944825;
            } else {
              result[0] += -0.0195438;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 96))) {
              result[0] += -0.032838166;
            } else {
              result[0] += -0.013822776;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 298))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 232))) {
              result[0] += -0.034293767;
            } else {
              result[0] += -0.010428538;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 38))) {
              result[0] += -0.008245598;
            } else {
              result[0] += 0.0061710696;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 80))) {
              result[0] += -0.00058508094;
            } else {
              result[0] += -0.048794635;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
              result[0] += 0.014517671;
            } else {
              result[0] += 0.029143566;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 210))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 162))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 10))) {
              result[0] += -0.051883638;
            } else {
              result[0] += -0.121891774;
            }
          } else {
            result[0] += 0.006774173;
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 108))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 20))) {
              result[0] += -0.05183841;
            } else {
              result[0] += -0.023333719;
            }
          } else {
            result[0] += 0.041755553;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 128))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += -0.021229211;
            } else {
              result[0] += 0.023228422;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 76))) {
              result[0] += 0.0065774354;
            } else {
              result[0] += -0.0068485746;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 28))) {
            result[0] += 0.049541216;
          } else {
            result[0] += 0.21175821;
          }
        }
      }
    } else {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 298))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 10))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 292))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.1843134;
            } else {
              result[0] += 0.011620779;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 280))) {
              result[0] += -0.049533408;
            } else {
              result[0] += 0.019996347;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 272))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
              result[0] += 0.07229465;
            } else {
              result[0] += 0.019160887;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += 0.0056915763;
            } else {
              result[0] += 0.04898624;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 328))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 56))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 124))) {
              result[0] += 0.010618052;
            } else {
              result[0] += 0.039828878;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
              result[0] += 0.08413791;
            } else {
              result[0] += 0.05419789;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 128))) {
              result[0] += 0.04146382;
            } else {
              result[0] += 0.08484615;
            }
          } else {
            result[0] += 0.17491151;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 146))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 28))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 30))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
              result[0] += -0.14357229;
            } else {
              result[0] += -0.06613964;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
              result[0] += -0.07836775;
            } else {
              result[0] += 0.008682779;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 20))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
              result[0] += -0.085223936;
            } else {
              result[0] += -0.10550176;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 84))) {
              result[0] += -0.04918786;
            } else {
              result[0] += -0.0712194;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 18))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
              result[0] += -0.0485091;
            } else {
              result[0] += -0.07085284;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 154))) {
              result[0] += -0.04310399;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 168))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 60))) {
              result[0] += -0.03786544;
            } else {
              result[0] += -0.06086583;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 24))) {
              result[0] += -0.026798641;
            } else {
              result[0] += -0.0029160806;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 66))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 120))) {
              result[0] += -0.041853126;
            } else {
              result[0] += -0.067471944;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.10217434;
            } else {
              result[0] += -0.031203955;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 240))) {
              result[0] += -0.020556139;
            } else {
              result[0] += 0.0036511559;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
              result[0] += -0.0052104336;
            } else {
              result[0] += 0.022399493;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 266))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
              result[0] += -0.050368812;
            } else {
              result[0] += -0.015018652;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 98))) {
              result[0] += 0.050732918;
            } else {
              result[0] += -0.017873917;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
              result[0] += 0.16741006;
            } else {
              result[0] += 0.08613744;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
              result[0] += 0.02335409;
            } else {
              result[0] += -0.0018855215;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 234))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 200))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 172))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += 0.0073416415;
            } else {
              result[0] += 0.029692546;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 146))) {
              result[0] += -0.0061891414;
            } else {
              result[0] += -0.021139516;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 378))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 224))) {
              result[0] += -0.018264985;
            } else {
              result[0] += -0.0061430032;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += 0.024317574;
            } else {
              result[0] += -0.023863515;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 294))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 90))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.06744468;
            } else {
              result[0] += 0.2128239;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += 0.013560965;
            } else {
              result[0] += -0.010385789;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 156))) {
              result[0] += 0.036912605;
            } else {
              result[0] += 0.008162293;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
              result[0] += 0.07527434;
            } else {
              result[0] += 0.013765368;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 362))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 74))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 132))) {
              result[0] += -0.040987097;
            } else {
              result[0] += 0.009021326;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
              result[0] += 0.026009029;
            } else {
              result[0] += 0.08366104;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 372))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 34))) {
              result[0] += 0.064916424;
            } else {
              result[0] += 0.09441544;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 4))) {
              result[0] += 0.08905909;
            } else {
              result[0] += 0.020877834;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 318))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 126))) {
              result[0] += 0.02257039;
            } else {
              result[0] += -0.025986714;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 202))) {
              result[0] += 0.05338145;
            } else {
              result[0] += 0.15790914;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 180))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 220))) {
              result[0] += 0.018759893;
            } else {
              result[0] += 0.056597423;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 318))) {
              result[0] += 0.06810229;
            } else {
              result[0] += 0.16054772;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 156))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 48))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 42))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 180))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 18))) {
              result[0] += -0.06675883;
            } else {
              result[0] += -0.04909338;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 80))) {
              result[0] += -0.045537602;
            } else {
              result[0] += -0.021407653;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 102))) {
            result[0] += -0.036747407;
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 38))) {
              result[0] += -0.0055241673;
            } else {
              result[0] += 0.0064017037;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 8))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
              result[0] += -0.1232142;
            } else {
              result[0] += -0.08944999;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
              result[0] += -0.103527024;
            } else {
              result[0] += -0.06685146;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 8))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 110))) {
              result[0] += -0.087132424;
            } else {
              result[0] += -0.0155411465;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.007360445;
            } else {
              result[0] += -0.042419024;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 104))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 42))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 102))) {
              result[0] += -0.02354617;
            } else {
              result[0] += 0.0067315414;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 120))) {
              result[0] += -0.02450421;
            } else {
              result[0] += -0.058597732;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 32))) {
              result[0] += 0.023825407;
            } else {
              result[0] += 0.18043439;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 8))) {
              result[0] += -0.043022137;
            } else {
              result[0] += 0.029165251;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 222))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 208))) {
              result[0] += -0.07280388;
            } else {
              result[0] += -0.046989717;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += 0.016371334;
            } else {
              result[0] += -0.0062531256;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
              result[0] += -0.00938953;
            } else {
              result[0] += 0.011327556;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 98))) {
              result[0] += -0.057005074;
            } else {
              result[0] += -0.011778428;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 292))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 268))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 264))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 46))) {
              result[0] += 0.030093227;
            } else {
              result[0] += 0.11276569;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
              result[0] += 0.0048071328;
            } else {
              result[0] += -0.030283172;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 192))) {
              result[0] += 0.07801795;
            } else {
              result[0] += 0.010139103;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 100))) {
              result[0] += 0.015687512;
            } else {
              result[0] += 0.057320017;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 128))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 162))) {
              result[0] += 0.16135572;
            } else {
              result[0] += 0.10743537;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 316))) {
              result[0] += 0.0299298;
            } else {
              result[0] += -0.025237197;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 62))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
              result[0] += 0.01356712;
            } else {
              result[0] += 0.027944017;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 284))) {
              result[0] += 0.09368516;
            } else {
              result[0] += 0.00735568;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 212))) {
              result[0] += 0.06613142;
            } else {
              result[0] += 0.030379802;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 296))) {
              result[0] += 0.037492864;
            } else {
              result[0] += 0.08968493;
            }
          }
        } else {
          result[0] += 0.14285138;
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 240))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 122))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 294))) {
              result[0] += 0.10012319;
            } else {
              result[0] += 0.04607804;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 298))) {
              result[0] += 0.00882749;
            } else {
              result[0] += 0.034421023;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 188))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 0))) {
              result[0] += 0.020126143;
            } else {
              result[0] += 0.056154665;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 344))) {
              result[0] += 0.06664233;
            } else {
              result[0] += 0.028045407;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 124))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 28))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 56))) {
              result[0] += -0.08264944;
            } else {
              result[0] += -0.017300455;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 132))) {
              result[0] += -0.09584019;
            } else {
              result[0] += -0.03871246;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 68))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 8))) {
              result[0] += -0.027366564;
            } else {
              result[0] += 0.005001399;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 10))) {
              result[0] += -0.015287772;
            } else {
              result[0] += -0.06076094;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
              result[0] += -0.01928839;
            } else {
              result[0] += -0.05802682;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 14))) {
              result[0] += -0.069114976;
            } else {
              result[0] += -0.050886452;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 168))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 58))) {
              result[0] += -0.02828979;
            } else {
              result[0] += -0.049417857;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 54))) {
              result[0] += 0.018446451;
            } else {
              result[0] += -0.017286932;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 116))) {
              result[0] += -0;
            } else {
              result[0] += -0.1095063;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
              result[0] += -0.112913184;
            } else {
              result[0] += -0.041024093;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 236))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
              result[0] += -0.036767628;
            } else {
              result[0] += -0.0056491303;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 50))) {
              result[0] += -0.05497813;
            } else {
              result[0] += -0.02247507;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 124))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 94))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 70))) {
              result[0] += -0.04725503;
            } else {
              result[0] += -0.022799945;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
              result[0] += -0.03265686;
            } else {
              result[0] += 0.0007802085;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 54))) {
              result[0] += 0.008465467;
            } else {
              result[0] += 0.022264598;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
              result[0] += -0.008557126;
            } else {
              result[0] += -0.050682377;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 230))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 56))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 84))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
              result[0] += -0.23723964;
            } else {
              result[0] += -0.0674392;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 96))) {
              result[0] += 0.017011134;
            } else {
              result[0] += -0.023735067;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 156))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
              result[0] += -0.021249495;
            } else {
              result[0] += 0.0016016475;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 86))) {
              result[0] += 0.004315733;
            } else {
              result[0] += 0.029064924;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 30))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 18))) {
              result[0] += 0.016252542;
            } else {
              result[0] += 0.07114914;
            }
          } else {
            result[0] += 0.18867145;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 4))) {
            result[0] += -0.043993976;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 52))) {
              result[0] += 0.04012975;
            } else {
              result[0] += 0.0784916;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 312))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 130))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 68))) {
            result[0] += -0.1251441;
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 124))) {
              result[0] += 0.1326;
            } else {
              result[0] += 0.07419976;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
              result[0] += 0.070240654;
            } else {
              result[0] += 0.0142424395;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += 0.008635036;
            } else {
              result[0] += 0.05565952;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 104))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 326))) {
              result[0] += 0.0021902567;
            } else {
              result[0] += 0.12992251;
            }
          } else {
            result[0] += 0.14213897;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 342))) {
              result[0] += 0.018112415;
            } else {
              result[0] += 0.048771314;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
              result[0] += 0.040432267;
            } else {
              result[0] += 0.06690516;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 158))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 48))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 4))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
              result[0] += -0.084643945;
            } else {
              result[0] += -0.05658353;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
              result[0] += -0.14270312;
            } else {
              result[0] += -0.096725866;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 10))) {
              result[0] += -0.072900295;
            } else {
              result[0] += -0.050103154;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.0026472413;
            } else {
              result[0] += -0.034194175;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 210))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 172))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
              result[0] += -0.08699354;
            } else {
              result[0] += -0.029962752;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 258))) {
              result[0] += -0.012024823;
            } else {
              result[0] += -0.040718153;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 38))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 278))) {
              result[0] += 1.4413594e-05;
            } else {
              result[0] += -0.01927815;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 52))) {
              result[0] += -0.004854994;
            } else {
              result[0] += 0.0106780855;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 88))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 28))) {
              result[0] += -0.07230939;
            } else {
              result[0] += -0.029629577;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
              result[0] += -0.018495014;
            } else {
              result[0] += 0.019453213;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 90))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
              result[0] += -0.01461862;
            } else {
              result[0] += 0.10096019;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
              result[0] += -0.029421015;
            } else {
              result[0] += 9.543482e-05;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 90))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 60))) {
              result[0] += -0.021362139;
            } else {
              result[0] += -0.063853346;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 62))) {
              result[0] += 0.00073504547;
            } else {
              result[0] += 0.019957332;
            }
          }
        } else {
          result[0] += 0.09791734;
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 294))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 60))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 18))) {
              result[0] += -0.005831562;
            } else {
              result[0] += -0.06499735;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 132))) {
              result[0] += -0.0071320483;
            } else {
              result[0] += -0.055945735;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 58))) {
              result[0] += 0.0037681882;
            } else {
              result[0] += 0.05542149;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 82))) {
              result[0] += 0.011588092;
            } else {
              result[0] += -0.0017960105;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 2))) {
              result[0] += -0.16209017;
            } else {
              result[0] += 0.019222626;
            }
          } else {
            result[0] += 0.13090767;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 270))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 14))) {
              result[0] += -0.030162899;
            } else {
              result[0] += -0.008859159;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 184))) {
              result[0] += 0.028086677;
            } else {
              result[0] += 0.0010727844;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 110))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 44))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
              result[0] += 0.07769736;
            } else {
              result[0] += 0.114956;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 62))) {
              result[0] += 0.026280258;
            } else {
              result[0] += 0.063517205;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 198))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 184))) {
              result[0] += 0.028841827;
            } else {
              result[0] += 0.07754768;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 220))) {
              result[0] += 0.023495244;
            } else {
              result[0] += 0.058550116;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 230))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 298))) {
              result[0] += 0.006005039;
            } else {
              result[0] += 0.0330562;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 182))) {
              result[0] += 0.02288873;
            } else {
              result[0] += 0.052012473;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 92))) {
              result[0] += -0.023153892;
            } else {
              result[0] += 0.06317775;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 122))) {
              result[0] += 0.09178891;
            } else {
              result[0] += 0.17105517;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 162))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 48))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 42))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 180))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 16))) {
              result[0] += -0.051588424;
            } else {
              result[0] += -0.03475411;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 240))) {
              result[0] += -0.0234789;
            } else {
              result[0] += 0.0038708851;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 214))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
              result[0] += -0.008086841;
            } else {
              result[0] += -0.027431283;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 68))) {
              result[0] += -0.001538304;
            } else {
              result[0] += 0.0073204334;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 2))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 56))) {
              result[0] += -0.05180776;
            } else {
              result[0] += 0.005863587;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 114))) {
              result[0] += -0.09820559;
            } else {
              result[0] += -0.042626407;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 110))) {
              result[0] += -0.048510984;
            } else {
              result[0] += -0.071686305;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 58))) {
              result[0] += -0.04765085;
            } else {
              result[0] += -0.010419096;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 28))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 94))) {
            result[0] += 0.06322976;
          } else {
            result[0] += -0.03347231;
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 12))) {
              result[0] += -0.17174184;
            } else {
              result[0] += -0.08866473;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
              result[0] += -0.019928426;
            } else {
              result[0] += -0.07354166;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
              result[0] += -0.020242494;
            } else {
              result[0] += -0.040445287;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
              result[0] += -0.008752908;
            } else {
              result[0] += 0.09380058;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 176))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 198))) {
              result[0] += 0.017689208;
            } else {
              result[0] += -0.0005958941;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 2))) {
              result[0] += -0.0046095774;
            } else {
              result[0] += -0.038875457;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 272))) {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
          result[0] += -0.046054542;
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 132))) {
            result[0] += -0.21343382;
          } else {
            result[0] += -0.087919936;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 50))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += 0.012674006;
            } else {
              result[0] += 0.11185076;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 308))) {
              result[0] += 0.041192535;
            } else {
              result[0] += 0.070588894;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 124))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 246))) {
              result[0] += 0.0028114773;
            } else {
              result[0] += 0.019441808;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 90))) {
              result[0] += 0.011045033;
            } else {
              result[0] += -0.022086669;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 20))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 306))) {
              result[0] += 0.071599804;
            } else {
              result[0] += 0.03955371;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 130))) {
              result[0] += 0.010970717;
            } else {
              result[0] += 0.03211321;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 334))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 68))) {
              result[0] += 0.059979778;
            } else {
              result[0] += 0.036047976;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 66))) {
              result[0] += 0.10536631;
            } else {
              result[0] += 0.024781441;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 322))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 254))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 210))) {
              result[0] += -0.052476525;
            } else {
              result[0] += -0.019687535;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
              result[0] += 0.012497628;
            } else {
              result[0] += -0.011794323;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 362))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 120))) {
              result[0] += 0.013189244;
            } else {
              result[0] += 0.03505987;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 372))) {
              result[0] += 0.056928415;
            } else {
              result[0] += 0.022988325;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 126))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 22))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 56))) {
              result[0] += -0.058271427;
            } else {
              result[0] += 0.0006807263;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
              result[0] += -0.10462822;
            } else {
              result[0] += -0.068169676;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 12))) {
              result[0] += -0.0072206096;
            } else {
              result[0] += 0.0307555;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 50))) {
              result[0] += -0.050685406;
            } else {
              result[0] += -0.023581887;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 296))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
              result[0] += -0.005921537;
            } else {
              result[0] += -0.042507444;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += -0.052036572;
            } else {
              result[0] += -0.034811404;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
            result[0] += 0.0008109753;
          } else {
            result[0] += -0.023712752;
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 22))) {
              result[0] += -0.081094466;
            } else {
              result[0] += -0.017712194;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
              result[0] += -0.09813706;
            } else {
              result[0] += -0.03229853;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 236))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 82))) {
              result[0] += -0.028393775;
            } else {
              result[0] += -0.0027526347;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 290))) {
              result[0] += -0.047838878;
            } else {
              result[0] += -0.026909484;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 18))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 2))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 12))) {
              result[0] += -0.13414939;
            } else {
              result[0] += -0.05853174;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 8))) {
              result[0] += -0.08385962;
            } else {
              result[0] += -0.029232157;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 112))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 94))) {
              result[0] += -0.012187723;
            } else {
              result[0] += -0.00030637285;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
              result[0] += -0.05476123;
            } else {
              result[0] += -0.002212812;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 234))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 62))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 88))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 56))) {
              result[0] += 0.034629773;
            } else {
              result[0] += 0.090757936;
            }
          } else {
            result[0] += -0.015683705;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 4))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += 0.0014823362;
            } else {
              result[0] += -0.11727782;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 328))) {
              result[0] += -0.011885715;
            } else {
              result[0] += 0.06458332;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += 0.009296111;
            } else {
              result[0] += -0.0022434364;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 390))) {
              result[0] += 0.09074904;
            } else {
              result[0] += 0.022570081;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.010527322;
            } else {
              result[0] += 0.0112762675;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 170))) {
              result[0] += -0.026221529;
            } else {
              result[0] += -0.0167725;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 128))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 156))) {
            result[0] += 0.134217;
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 242))) {
              result[0] += 0.055591363;
            } else {
              result[0] += 0.09201013;
            }
          }
        } else {
          result[0] += -0.07688349;
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 298))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 10))) {
              result[0] += 0.008139768;
            } else {
              result[0] += 0.027951911;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
              result[0] += 0.03482144;
            } else {
              result[0] += 0.12041791;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 330))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 56))) {
              result[0] += 0.019060235;
            } else {
              result[0] += 0.03520998;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 268))) {
              result[0] += 0.05164655;
            } else {
              result[0] += 0.10352349;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 166))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 62))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 4))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 6))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
              result[0] += -0.05586126;
            } else {
              result[0] += -0.0062299026;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 2))) {
              result[0] += -0.08666905;
            } else {
              result[0] += -0.057932485;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 106))) {
              result[0] += -0.037010778;
            } else {
              result[0] += -0.060017806;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.008048224;
            } else {
              result[0] += -0.025649264;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 180))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.018599523;
            } else {
              result[0] += -0.063888915;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 8))) {
              result[0] += -0.055958636;
            } else {
              result[0] += 0.0054995283;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 242))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 60))) {
              result[0] += -0.015452884;
            } else {
              result[0] += -0.0031880846;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
              result[0] += 6.0049097e-06;
            } else {
              result[0] += 0.011630835;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 80))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 104))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.012825543;
            } else {
              result[0] += 0.0061619915;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += -0.035948735;
            } else {
              result[0] += -0.005659299;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 22))) {
              result[0] += -0.0034629384;
            } else {
              result[0] += 0.11748111;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 14))) {
              result[0] += -0.045964625;
            } else {
              result[0] += 0.027189994;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 220))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 288))) {
              result[0] += -0.043134686;
            } else {
              result[0] += -0.008114795;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 136))) {
              result[0] += 0.022731418;
            } else {
              result[0] += -0.0039848676;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
              result[0] += -0.0025353373;
            } else {
              result[0] += 0.017748738;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 68))) {
              result[0] += -0.00209215;
            } else {
              result[0] += -0.026470816;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 292))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 268))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            result[0] += -0.0426252;
          } else {
            result[0] += -0.17894174;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 46))) {
              result[0] += 0.034646492;
            } else {
              result[0] += 0.08830755;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
              result[0] += 0.0036898192;
            } else {
              result[0] += 0.10474273;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 130))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 278))) {
              result[0] += 0.0591173;
            } else {
              result[0] += 0.10110114;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 316))) {
              result[0] += 0.017330313;
            } else {
              result[0] += -0.03201217;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 92))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 86))) {
              result[0] += 0.020424707;
            } else {
              result[0] += 0.06761751;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 374))) {
              result[0] += 0.0006410443;
            } else {
              result[0] += 0.025116405;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 342))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 228))) {
              result[0] += 0.002580609;
            } else {
              result[0] += 0.015501295;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 26))) {
              result[0] += -0.021991013;
            } else {
              result[0] += 0.1534664;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 272))) {
              result[0] += -0.004891763;
            } else {
              result[0] += 0.024440682;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 344))) {
              result[0] += 0.04597979;
            } else {
              result[0] += 0.013181034;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
          result[0] += 0.08511517;
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 12))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 44))) {
              result[0] += 0.017548429;
            } else {
              result[0] += -0.051454138;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 312))) {
              result[0] += 0.0665479;
            } else {
              result[0] += 0.03926411;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 52))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 4))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 120))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += -0.034856033;
            } else {
              result[0] += 0.035472337;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.082092844;
            } else {
              result[0] += -0.057657477;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 122))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 96))) {
              result[0] += 0.042515777;
            } else {
              result[0] += -0.018864583;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 142))) {
              result[0] += -0.057759553;
            } else {
              result[0] += -0.020723581;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 114))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 26))) {
              result[0] += 0.054745015;
            } else {
              result[0] += 0.017524384;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 164))) {
              result[0] += -0.03406791;
            } else {
              result[0] += -0.08434524;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 28))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 142))) {
              result[0] += 0.017600825;
            } else {
              result[0] += -0.03500613;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 18))) {
              result[0] += -0.046339255;
            } else {
              result[0] += -0.018127348;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 148))) {
            result[0] += -0.037658464;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 6))) {
              result[0] += 0.0075216624;
            } else {
              result[0] += 0.0364659;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 134))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 102))) {
              result[0] += 0.033616688;
            } else {
              result[0] += 0.083399825;
            }
          } else {
            result[0] += 0.12963086;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.074191265;
            } else {
              result[0] += 0.007478449;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 42))) {
              result[0] += 0.098748855;
            } else {
              result[0] += 0.030946104;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
              result[0] += -0.026153265;
            } else {
              result[0] += -0.15095265;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 74))) {
              result[0] += -0.018939627;
            } else {
              result[0] += -0.06403255;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
            result[0] += -0.021922093;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 134))) {
              result[0] += -0.08688069;
            } else {
              result[0] += -0.033859696;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 86))) {
              result[0] += -0.016842697;
            } else {
              result[0] += -0.0006648888;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 242))) {
              result[0] += -0.023793818;
            } else {
              result[0] += -0.035974815;
            }
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 302))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 250))) {
              result[0] += 0.012690376;
            } else {
              result[0] += -0.0033918878;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
              result[0] += -0.005548475;
            } else {
              result[0] += -0.036335457;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 206))) {
            result[0] += -0.0005064619;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 376))) {
              result[0] += 0.03696196;
            } else {
              result[0] += 0.015889136;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 86))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 84))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 44))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 108))) {
              result[0] += -0.019162795;
            } else {
              result[0] += -0.0054707523;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.0037607402;
            } else {
              result[0] += -0.009698882;
            }
          }
        } else {
          result[0] += -0.069641024;
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 256))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 192))) {
              result[0] += 0.045991335;
            } else {
              result[0] += 0.0303468;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += 0.018556785;
            } else {
              result[0] += -0.028926486;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 222))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 126))) {
              result[0] += 0.014895338;
            } else {
              result[0] += -0.0033647448;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += 0.031980734;
            } else {
              result[0] += 0.0069378153;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 28))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 30))) {
              result[0] += -0.091816425;
            } else {
              result[0] += -0.053907175;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.035644125;
            } else {
              result[0] += -0.08623389;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += -0.038305838;
            } else {
              result[0] += -0.022053199;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 34))) {
              result[0] += -0.041282233;
            } else {
              result[0] += -0.011869354;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 136))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 22))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 118))) {
              result[0] += -0.008556094;
            } else {
              result[0] += -0.023536053;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 40))) {
              result[0] += -0.023365794;
            } else {
              result[0] += -0.0052509154;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 94))) {
            result[0] += -0.008722498;
          } else {
            result[0] += -0.08987408;
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
              result[0] += 0.032462854;
            } else {
              result[0] += -0.013032225;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 44))) {
              result[0] += 0.0020141487;
            } else {
              result[0] += 0.016696744;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 172))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 46))) {
              result[0] += 0.0063471817;
            } else {
              result[0] += 0.01982216;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
              result[0] += 0.00396263;
            } else {
              result[0] += -0.017814338;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 28))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 310))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 22))) {
              result[0] += -0.042612687;
            } else {
              result[0] += -0.0046601035;
            }
          } else {
            result[0] += -0.007730845;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 66))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 48))) {
              result[0] += -0.039969202;
            } else {
              result[0] += -0.116455294;
            }
          } else {
            result[0] += -0.02463033;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 208))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 60))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 144))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 42))) {
              result[0] += 0.022728456;
            } else {
              result[0] += -0.023419065;
            }
          } else {
            result[0] += 0.06361882;
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 22))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 74))) {
              result[0] += -0.025523862;
            } else {
              result[0] += -0.051117957;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.006611984;
            } else {
              result[0] += -0.04705615;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += 0.010768503;
            } else {
              result[0] += -0.0008824992;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 298))) {
              result[0] += -0.040758982;
            } else {
              result[0] += -0.008637625;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 130))) {
            result[0] += 0.029029582;
          } else {
            result[0] += 0.08626362;
          }
        }
      }
    } else {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 302))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
              result[0] += 0.011424046;
            } else {
              result[0] += 0.04683126;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 72))) {
              result[0] += -0.033311624;
            } else {
              result[0] += 0.076620966;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 264))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 106))) {
              result[0] += -0.01699209;
            } else {
              result[0] += -0.032285534;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 104))) {
              result[0] += 0.008964994;
            } else {
              result[0] += -0.01650845;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 60))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 316))) {
              result[0] += 0.025268046;
            } else {
              result[0] += 0.055473793;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 66))) {
              result[0] += 0.08008851;
            } else {
              result[0] += 0.01032548;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 326))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 138))) {
              result[0] += 0.0054397047;
            } else {
              result[0] += 0.02308716;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
              result[0] += 0.08063211;
            } else {
              result[0] += 0.03532891;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 166))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 48))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 10))) {
              result[0] += -0.04106638;
            } else {
              result[0] += -0.02793704;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
              result[0] += 0.022730371;
            } else {
              result[0] += -0.015745891;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 38))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += -0.056902584;
            } else {
              result[0] += -0.11115749;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 52))) {
              result[0] += -0.0027035407;
            } else {
              result[0] += -0.0484084;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 64))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
            result[0] += -0.06669981;
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 8))) {
              result[0] += -0.058218993;
            } else {
              result[0] += -0.01607107;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 276))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 240))) {
              result[0] += -0.004913365;
            } else {
              result[0] += 0.010270695;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 288))) {
              result[0] += -0.013211744;
            } else {
              result[0] += -0.029026702;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 84))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 96))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
              result[0] += -0.017454801;
            } else {
              result[0] += 0.0241193;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.07878145;
            } else {
              result[0] += -0.021389706;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 150))) {
              result[0] += 0.005198505;
            } else {
              result[0] += -0.025203273;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
              result[0] += -0.0036238257;
            } else {
              result[0] += 0.030112168;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 206))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 6))) {
              result[0] += -0.014281181;
            } else {
              result[0] += 0.044825673;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
              result[0] += 0.00095986744;
            } else {
              result[0] += 0.014824574;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 78))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 248))) {
              result[0] += -0.004780053;
            } else {
              result[0] += 0.023393003;
            }
          } else {
            result[0] += -0.05524193;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 240))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 264))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 200))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
              result[0] += 0.008043467;
            } else {
              result[0] += -0.035642993;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
              result[0] += 0.025907112;
            } else {
              result[0] += 0.0037340687;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 164))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 90))) {
              result[0] += 0.0053138766;
            } else {
              result[0] += -0.019315336;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
              result[0] += 0.012754604;
            } else {
              result[0] += -0.017428217;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 316))) {
              result[0] += -0.018819278;
            } else {
              result[0] += 0.005015341;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 322))) {
              result[0] += -0.046993464;
            } else {
              result[0] += -0.00060153054;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 334))) {
              result[0] += 0.032121107;
            } else {
              result[0] += 0.07720378;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 206))) {
              result[0] += -0.01248775;
            } else {
              result[0] += 0.021081064;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 220))) {
              result[0] += -0.002822095;
            } else {
              result[0] += 0.010395023;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 24))) {
              result[0] += -0.00028166902;
            } else {
              result[0] += 0.05817884;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 368))) {
              result[0] += 0.033362225;
            } else {
              result[0] += 0.059052836;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 76))) {
              result[0] += -0.022754088;
            } else {
              result[0] += 0.032660067;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 164))) {
              result[0] += 0.05608514;
            } else {
              result[0] += 0.028797118;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 310))) {
              result[0] += 0.0028316458;
            } else {
              result[0] += 0.023484427;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 308))) {
              result[0] += 0.029160103;
            } else {
              result[0] += 0.06451139;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 212))) {
              result[0] += -0.008154633;
            } else {
              result[0] += 0.02224008;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 102))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 66))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 34))) {
              result[0] += -0.025441715;
            } else {
              result[0] += -0;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.09877158;
            } else {
              result[0] += -0.05378988;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 110))) {
              result[0] += -0.02562209;
            } else {
              result[0] += -0.042403318;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
              result[0] += 0.02021136;
            } else {
              result[0] += -0.01660019;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 28))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
            result[0] += 0.01719434;
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 10))) {
              result[0] += -0.018652868;
            } else {
              result[0] += -0.072789736;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.0050430708;
            } else {
              result[0] += -0.020398756;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 88))) {
              result[0] += 0.0035772503;
            } else {
              result[0] += 0.04239833;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 222))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
              result[0] += -0.008277955;
            } else {
              result[0] += -0.02545519;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
              result[0] += 0.035175905;
            } else {
              result[0] += -0.0068374695;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 274))) {
              result[0] += 0.0034994117;
            } else {
              result[0] += -0.012190169;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
              result[0] += -0.0075547704;
            } else {
              result[0] += 0.01670718;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 130))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 100))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 46))) {
              result[0] += -0.043555766;
            } else {
              result[0] += -0.09016201;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 176))) {
              result[0] += -0.043037914;
            } else {
              result[0] += -2.9695482e-05;
            }
          }
        } else {
          result[0] += -0.024823932;
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 230))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 6))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 120))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            result[0] += 0.05400118;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 80))) {
              result[0] += -0.07639449;
            } else {
              result[0] += -0.016434733;
            }
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 148))) {
              result[0] += -0.05022436;
            } else {
              result[0] += 0.00233022;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 126))) {
              result[0] += -0.055654902;
            } else {
              result[0] += -0.13723731;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 104))) {
              result[0] += -0.003547279;
            } else {
              result[0] += 0.0030764283;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
              result[0] += 0.016396696;
            } else {
              result[0] += 0.07968046;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
              result[0] += 0.017225603;
            } else {
              result[0] += 0.037653413;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.077591464;
            } else {
              result[0] += 0.15476403;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 130))) {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 238))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 298))) {
              result[0] += -0.016144833;
            } else {
              result[0] += 0.032200087;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 282))) {
              result[0] += 0.06538327;
            } else {
              result[0] += 0.10325643;
            }
          }
        } else {
          result[0] += -0.10228916;
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 272))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
              result[0] += -0.008800676;
            } else {
              result[0] += 0.009970379;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
              result[0] += -0.015078469;
            } else {
              result[0] += -0.03748675;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
              result[0] += 0.034970764;
            } else {
              result[0] += -0.026505647;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 88))) {
              result[0] += 0.037449267;
            } else {
              result[0] += 0.012034763;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 160))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 64))) {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 32))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 162))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += -0.042211335;
            } else {
              result[0] += -0.0153662665;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.08464201;
            } else {
              result[0] += -0.039633475;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 18))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 32))) {
              result[0] += -0.0023176766;
            } else {
              result[0] += 0.011839011;
            }
          } else {
            result[0] += -0.024568608;
          }
        }
      } else {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 114))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 50))) {
              result[0] += -0.011506539;
            } else {
              result[0] += -0.041527543;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 30))) {
              result[0] += 0.006311471;
            } else {
              result[0] += 0.096107006;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 20))) {
              result[0] += -0.07760205;
            } else {
              result[0] += -0.024386143;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
              result[0] += 0.0042932476;
            } else {
              result[0] += -0.024321787;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 278))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 108))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 78))) {
              result[0] += -0.00662189;
            } else {
              result[0] += 0.01475962;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 130))) {
              result[0] += 0.03366244;
            } else {
              result[0] += 0.01162212;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 114))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 28))) {
              result[0] += 0.030401085;
            } else {
              result[0] += 0.1034078;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 76))) {
              result[0] += 0.06511612;
            } else {
              result[0] += 0.004562137;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 258))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 132))) {
              result[0] += -0.017277284;
            } else {
              result[0] += -0.00040395462;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 274))) {
              result[0] += -0.028246973;
            } else {
              result[0] += -0.043952625;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 212))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
              result[0] += -0.045117795;
            } else {
              result[0] += -0.00930117;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 90))) {
              result[0] += 0.013196451;
            } else {
              result[0] += -0.019043565;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 310))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 42))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 84))) {
            result[0] += -0.026572356;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 164))) {
              result[0] += 0.051293522;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 88))) {
            result[0] += 0.08133901;
          } else {
            result[0] += 0.05325532;
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 114))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 6))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 248))) {
              result[0] += -0.015609048;
            } else {
              result[0] += -0.038890243;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 102))) {
              result[0] += -0.008053626;
            } else {
              result[0] += 0.0021804231;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
              result[0] += 0.013441979;
            } else {
              result[0] += 0.0015576767;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 48))) {
              result[0] += -0.012684277;
            } else {
              result[0] += 0.0018506636;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 66))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 146))) {
            result[0] += 0.1326864;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
              result[0] += 0.0802758;
            } else {
              result[0] += 0.04734662;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 162))) {
              result[0] += -0.0053350735;
            } else {
              result[0] += -0.040489998;
            }
          } else {
            result[0] += 0.057940014;
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 228))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 318))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 38))) {
              result[0] += -0.00303559;
            } else {
              result[0] += 0.027338777;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 98))) {
              result[0] += 0.021263791;
            } else {
              result[0] += 0.0065736147;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 220))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 294))) {
              result[0] += 0.036004547;
            } else {
              result[0] += 0.00041290582;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
              result[0] += 0.003415088;
            } else {
              result[0] += 0.03065363;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 112))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 16))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 182))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 30))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
              result[0] += -0.061424643;
            } else {
              result[0] += -0.004366158;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 12))) {
              result[0] += -0.029432973;
            } else {
              result[0] += 0.024416348;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 44))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 38))) {
              result[0] += -0.028231425;
            } else {
              result[0] += 0.013277724;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 46))) {
              result[0] += -0.05933566;
            } else {
              result[0] += -0.033664998;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 90))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 64))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 276))) {
              result[0] += 0.0039472985;
            } else {
              result[0] += -0.010128739;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 96))) {
              result[0] += -0.032275613;
            } else {
              result[0] += -0.007163971;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 148))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
              result[0] += -0.009553356;
            } else {
              result[0] += -0.024487672;
            }
          } else {
            result[0] += -0.04257847;
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 106))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
              result[0] += -0.017238883;
            } else {
              result[0] += -0.005016202;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
              result[0] += -0.031886365;
            } else {
              result[0] += -0.013730754;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 64))) {
              result[0] += 0.008240308;
            } else {
              result[0] += 0.03919262;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 38))) {
              result[0] += 0.005747178;
            } else {
              result[0] += -0.058204617;
            }
          }
        }
      } else {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 268))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
              result[0] += -0.055713423;
            } else {
              result[0] += -0.126311;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 160))) {
              result[0] += -0.010013968;
            } else {
              result[0] += -0.03667828;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 62))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 90))) {
              result[0] += -0.021773633;
            } else {
              result[0] += -0.060646553;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 76))) {
              result[0] += -0.028524075;
            } else {
              result[0] += -0.0014835782;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 240))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 250))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 58))) {
              result[0] += -0.009143605;
            } else {
              result[0] += 0.0004789403;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
              result[0] += 0.017569097;
            } else {
              result[0] += 0.06969844;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.053881604;
            } else {
              result[0] += 0.030100072;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 28))) {
              result[0] += 0.07642787;
            } else {
              result[0] += 0.14859438;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 276))) {
              result[0] += -0.0064661647;
            } else {
              result[0] += 0.0021556418;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
              result[0] += -0.042718846;
            } else {
              result[0] += 0.004742034;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 96))) {
              result[0] += 0.0150559945;
            } else {
              result[0] += 0.02904155;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 196))) {
              result[0] += 0.030410746;
            } else {
              result[0] += -0.00021071863;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 130))) {
              result[0] += -0.0019750185;
            } else {
              result[0] += 0.010826028;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 92))) {
              result[0] += -0.027783621;
            } else {
              result[0] += 0.040064003;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 34))) {
              result[0] += 0.025106177;
            } else {
              result[0] += 0.04847647;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
              result[0] += -0.020649737;
            } else {
              result[0] += 0.047101017;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 110))) {
              result[0] += 0.0425664;
            } else {
              result[0] += 0.010982556;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 320))) {
              result[0] += 0.0027861602;
            } else {
              result[0] += 0.024253966;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 268))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
              result[0] += 0.040306833;
            } else {
              result[0] += 0.019420182;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 250))) {
              result[0] += -0.07389223;
            } else {
              result[0] += -0.018163418;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 166))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 76))) {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 32))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 134))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.033120375;
            } else {
              result[0] += -0.019771839;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
              result[0] += 0.033069532;
            } else {
              result[0] += -0.024330398;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
            result[0] += -0.00891195;
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 260))) {
              result[0] += -0.06895655;
            } else {
              result[0] += -0.029440561;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 112))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
              result[0] += -0.008321228;
            } else {
              result[0] += -0.032401457;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 28))) {
              result[0] += 0.01681263;
            } else {
              result[0] += -0.011843439;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
              result[0] += 0.009037083;
            } else {
              result[0] += 0.084029466;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
              result[0] += -0.039284047;
            } else {
              result[0] += 0.0024618404;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 30))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 138))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 284))) {
              result[0] += -0.004148552;
            } else {
              result[0] += -0.028972754;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 182))) {
              result[0] += -0.060149025;
            } else {
              result[0] += -0.023092858;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 246))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 124))) {
              result[0] += -0.0048736627;
            } else {
              result[0] += 0.011900749;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 152))) {
              result[0] += -0.027425656;
            } else {
              result[0] += -0.0049680914;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
              result[0] += 0.018673597;
            } else {
              result[0] += -0.002092194;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 158))) {
              result[0] += 0.025586123;
            } else {
              result[0] += 0.043332465;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 140))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 134))) {
              result[0] += -0.009704775;
            } else {
              result[0] += 0.033585932;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 130))) {
              result[0] += -0.04070072;
            } else {
              result[0] += -0.02585305;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
        if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 0))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 132))) {
              result[0] += 0.026704356;
            } else {
              result[0] += -0.0026301623;
            }
          } else {
            result[0] += -0.083959244;
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 116))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.00727385;
            } else {
              result[0] += -0.00793132;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 34))) {
              result[0] += -0.0025618703;
            } else {
              result[0] += 0.03066427;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 80))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 192))) {
              result[0] += -0.0036948516;
            } else {
              result[0] += 0.014927791;
            }
          } else {
            result[0] += 0.06770062;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 112))) {
              result[0] += 0.021252338;
            } else {
              result[0] += 0.08691488;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 110))) {
              result[0] += 0.030034242;
            } else {
              result[0] += 0.0126524735;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 84))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 146))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 294))) {
              result[0] += 0.03448388;
            } else {
              result[0] += 0.012207389;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 82))) {
              result[0] += -0.011361829;
            } else {
              result[0] += 0.010800744;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 374))) {
            result[0] += -0.019659309;
          } else {
            result[0] += -0.010299957;
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 222))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 128))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
              result[0] += -0.022616882;
            } else {
              result[0] += 0.0058049723;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 210))) {
              result[0] += -0.05356568;
            } else {
              result[0] += -0.011999037;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
              result[0] += 0.007046343;
            } else {
              result[0] += 0.04422299;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 200))) {
              result[0] += -0.023396257;
            } else {
              result[0] += -0.015149494;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 110))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 16))) {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 4))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 38))) {
            result[0] += -0.012656443;
          } else {
            result[0] += 0.04053043;
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 30))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
              result[0] += -0.04405209;
            } else {
              result[0] += -0.07002534;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
              result[0] += 0.0050658057;
            } else {
              result[0] += -0.035065096;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 36))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 44))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 40))) {
              result[0] += -0.016180787;
            } else {
              result[0] += 0.035941508;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 62))) {
              result[0] += 0.0028223798;
            } else {
              result[0] += -0.028867302;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 96))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 60))) {
              result[0] += -0.0008212989;
            } else {
              result[0] += -0.019350568;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 100))) {
              result[0] += -0.012334666;
            } else {
              result[0] += -0.02596775;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 110))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
              result[0] += -0.03201983;
            } else {
              result[0] += -0.009261032;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.043002192;
            } else {
              result[0] += -0.004841908;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 192))) {
              result[0] += -0.034105346;
            } else {
              result[0] += -0.001769832;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 18))) {
              result[0] += -0.026078701;
            } else {
              result[0] += -0.0034125247;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 68))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 204))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 72))) {
              result[0] += -0.035768237;
            } else {
              result[0] += 0.007326287;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 84))) {
              result[0] += -0.013266384;
            } else {
              result[0] += 0.0073778555;
            }
          }
        } else {
          result[0] += 0.04042886;
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 262))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 314))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
              result[0] += -0.031073779;
            } else {
              result[0] += 0.009902022;
            }
          } else {
            result[0] += -0.10413245;
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 304))) {
              result[0] += -0.0015145864;
            } else {
              result[0] += 0.0044673397;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
              result[0] += 0.016366184;
            } else {
              result[0] += 0.06682373;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
          result[0] += 0.1125614;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 158))) {
              result[0] += 0.015199658;
            } else {
              result[0] += -0.047179397;
            }
          } else {
            result[0] += 0.07871269;
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 192))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 304))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 140))) {
              result[0] += 0.037271794;
            } else {
              result[0] += 0.055074245;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 350))) {
              result[0] += 0.01456887;
            } else {
              result[0] += 0.043336138;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 62))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
              result[0] += -0.011651535;
            } else {
              result[0] += 0.02064044;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 326))) {
              result[0] += 0.040328953;
            } else {
              result[0] += 0.016403003;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 12))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 122))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 262))) {
              result[0] += -0.0107279355;
            } else {
              result[0] += 0.005842201;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
              result[0] += -0.038425323;
            } else {
              result[0] += 0.021354888;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 342))) {
              result[0] += 0.045149498;
            } else {
              result[0] += -0;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 272))) {
              result[0] += 0.0030525408;
            } else {
              result[0] += 0.016887192;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 80))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 80))) {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 138))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 76))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
              result[0] += -0.022748465;
            } else {
              result[0] += -0.0015031536;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 58))) {
              result[0] += 0.038551223;
            } else {
              result[0] += -0.02340021;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 46))) {
            result[0] += -0.022547202;
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 64))) {
              result[0] += 0.006482385;
            } else {
              result[0] += 0.043839242;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 150))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 84))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
              result[0] += -0.032270882;
            } else {
              result[0] += 0.041581474;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 74))) {
              result[0] += -0.02748827;
            } else {
              result[0] += -0.06621357;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
            result[0] += -0.15229563;
          } else {
            result[0] += -0.056743264;
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 290))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 18))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += 0.0090522;
            } else {
              result[0] += -0.021619566;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 88))) {
              result[0] += 0.011768992;
            } else {
              result[0] += 0.034092333;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 174))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 296))) {
              result[0] += -0.0050243097;
            } else {
              result[0] += 0.0058637843;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 218))) {
              result[0] += -0.022230322;
            } else {
              result[0] += -0;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 234))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 178))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += -0.0040832614;
            } else {
              result[0] += -0.026436746;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 188))) {
              result[0] += 0.0104423;
            } else {
              result[0] += -0.009707513;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 146))) {
              result[0] += 0.021574568;
            } else {
              result[0] += -0.01677997;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.01483564;
            } else {
              result[0] += -0.044649884;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 266))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 270))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 230))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 58))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
              result[0] += 0.027206311;
            } else {
              result[0] += -0.0112104425;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.002884076;
            } else {
              result[0] += -0.008946375;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 242))) {
              result[0] += -0.021231398;
            } else {
              result[0] += -0.007831926;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
              result[0] += -0.010156151;
            } else {
              result[0] += 0.006795832;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += -0;
            } else {
              result[0] += 0.01808864;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 156))) {
              result[0] += 0.0849195;
            } else {
              result[0] += 0.040363234;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 208))) {
              result[0] += 0.018572433;
            } else {
              result[0] += -0.016821038;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 238))) {
              result[0] += -0.013755165;
            } else {
              result[0] += 0.010804783;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 124))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
              result[0] += 0.03828322;
            } else {
              result[0] += 0.0062655816;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 4))) {
              result[0] += -0.039595883;
            } else {
              result[0] += -0.00876256;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 162))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
              result[0] += 0.028006807;
            } else {
              result[0] += 0.011207984;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 104))) {
              result[0] += -0;
            } else {
              result[0] += 0.04828812;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 236))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 134))) {
              result[0] += 0.019871203;
            } else {
              result[0] += -0.0020931112;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 164))) {
              result[0] += -0.054860026;
            } else {
              result[0] += -0.016543666;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 126))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
              result[0] += 0.017333688;
            } else {
              result[0] += -0.014761999;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
              result[0] += 0.0047273724;
            } else {
              result[0] += 0.018399872;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 90))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 8))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
              result[0] += -0.07626998;
            } else {
              result[0] += 0.020595873;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += 0.0014158572;
            } else {
              result[0] += 0.039809644;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 26))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 28))) {
              result[0] += -0.047608785;
            } else {
              result[0] += -0.08177885;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
              result[0] += -0.018627753;
            } else {
              result[0] += -0.03965402;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 152))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += -0.016198985;
            } else {
              result[0] += -0.034763936;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
              result[0] += 0.0029702466;
            } else {
              result[0] += -0.010453141;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 80))) {
              result[0] += 0.002711076;
            } else {
              result[0] += -0.040787876;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
              result[0] += -0.0030517352;
            } else {
              result[0] += -0.018486565;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 204))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 296))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 34))) {
              result[0] += -0.01466268;
            } else {
              result[0] += 0.0016207602;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
              result[0] += 0.014294909;
            } else {
              result[0] += -0.00088145724;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 270))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 230))) {
              result[0] += 0.017696586;
            } else {
              result[0] += 0.0033850037;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 96))) {
              result[0] += 0.003758549;
            } else {
              result[0] += -0.008070237;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 108))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 38))) {
            result[0] += -0.012763503;
          } else {
            result[0] += 0.015098961;
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 260))) {
            result[0] += -0.06455403;
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 26))) {
              result[0] += -0.017203202;
            } else {
              result[0] += 0.0007284813;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 260))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 264))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
              result[0] += -0.0004502135;
            } else {
              result[0] += -0.013600699;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 104))) {
              result[0] += 0.0078685535;
            } else {
              result[0] += 0.052134164;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
              result[0] += -0.047534015;
            } else {
              result[0] += 0.021326948;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.056880232;
            } else {
              result[0] += 0.09747363;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 92))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 242))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 276))) {
              result[0] += 0.023308432;
            } else {
              result[0] += 0.011319171;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += 0.051132478;
            } else {
              result[0] += 0.02103916;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 238))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 288))) {
              result[0] += 0.006215774;
            } else {
              result[0] += -0.0061574043;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 330))) {
              result[0] += 0.0041284435;
            } else {
              result[0] += 0.023122046;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 192))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 288))) {
              result[0] += 0.034480914;
            } else {
              result[0] += 0.01336052;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.056260873;
            } else {
              result[0] += 0.042514727;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 172))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += 0.02588587;
            } else {
              result[0] += -0.032997902;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 316))) {
              result[0] += -0.0100605395;
            } else {
              result[0] += 0.020955171;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 12))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 122))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 320))) {
              result[0] += 0.0012241629;
            } else {
              result[0] += 0.010477929;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
              result[0] += -0.038784113;
            } else {
              result[0] += 0.019064387;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 308))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 128))) {
              result[0] += 0.014094616;
            } else {
              result[0] += -0.0027222696;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 342))) {
              result[0] += 0.043263398;
            } else {
              result[0] += 0.016619174;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 164))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 60))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 2))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 106))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 52))) {
            result[0] += -0.07700386;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
              result[0] += -0.027042255;
            } else {
              result[0] += 0.0029873038;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 180))) {
              result[0] += -0.063133374;
            } else {
              result[0] += -0.0046702675;
            }
          } else {
            result[0] += -0.097486384;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 114))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 110))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 258))) {
              result[0] += -0.011726489;
            } else {
              result[0] += -0.058912832;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 90))) {
              result[0] += -0.0659786;
            } else {
              result[0] += -0.02953866;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 58))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 90))) {
              result[0] += 0.0242778;
            } else {
              result[0] += -0.0023044557;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 118))) {
              result[0] += 0.029503396;
            } else {
              result[0] += -0.015513777;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 278))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 100))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 96))) {
              result[0] += -0.003784867;
            } else {
              result[0] += 0.012548617;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 26))) {
              result[0] += 0.0061970223;
            } else {
              result[0] += 0.023837453;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 330))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
              result[0] += 0.050173163;
            } else {
              result[0] += -0;
            }
          } else {
            result[0] += -0.007900632;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 86))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 24))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 288))) {
              result[0] += -0.027921408;
            } else {
              result[0] += -0.0038356972;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 126))) {
              result[0] += -0.0057223137;
            } else {
              result[0] += 0.010252192;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 188))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 232))) {
              result[0] += -0.0093339635;
            } else {
              result[0] += -0.002617002;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
              result[0] += -0.01492499;
            } else {
              result[0] += -0.025125777;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 308))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 42))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
              result[0] += 0.03968758;
            } else {
              result[0] += -0.001241333;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 6))) {
              result[0] += -0.01191957;
            } else {
              result[0] += 0.002430247;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 116))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 54))) {
              result[0] += -0;
            } else {
              result[0] += 0.08623359;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 168))) {
              result[0] += -0.0005277793;
            } else {
              result[0] += 0.01962237;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
              result[0] += -0.0047836783;
            } else {
              result[0] += -0.015227089;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
              result[0] += 0.008447981;
            } else {
              result[0] += 0.04540634;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 232))) {
            result[0] += -0.022486782;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
              result[0] += -0;
            } else {
              result[0] += -0.015179853;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 16))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 4))) {
              result[0] += -0.041306626;
            } else {
              result[0] += -0.008218481;
            }
          } else {
            result[0] += 0.042096373;
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 316))) {
              result[0] += 0.022573106;
            } else {
              result[0] += 0.07205055;
            }
          } else {
            result[0] += 0.028815215;
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 318))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 38))) {
              result[0] += -0.0054102;
            } else {
              result[0] += 0.019898819;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 98))) {
              result[0] += 0.013574933;
            } else {
              result[0] += 0.0023076402;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 294))) {
              result[0] += 0.023586577;
            } else {
              result[0] += -0.004840189;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
              result[0] += -0.0051436094;
            } else {
              result[0] += 0.020131659;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 162))) {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 84))) {
      if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 120))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 88))) {
              result[0] += -0.01574367;
            } else {
              result[0] += 0.0011124586;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 46))) {
              result[0] += -0.07613191;
            } else {
              result[0] += -0.01636207;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 74))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 24))) {
              result[0] += 0.014280135;
            } else {
              result[0] += 0.037306037;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 24))) {
              result[0] += -0.024091965;
            } else {
              result[0] += 0.0048261676;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 124))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 202))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
              result[0] += -0.04113116;
            } else {
              result[0] += -0.061350565;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 50))) {
              result[0] += -0.03143156;
            } else {
              result[0] += -0.0036942845;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 162))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 4))) {
              result[0] += 0.024413232;
            } else {
              result[0] += -0.01646574;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 46))) {
              result[0] += -0.016089493;
            } else {
              result[0] += 0.006952781;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 144))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 274))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 20))) {
              result[0] += -0.023807703;
            } else {
              result[0] += 0.0015813807;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 332))) {
              result[0] += 0.029507706;
            } else {
              result[0] += -0.009649085;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 36))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 24))) {
              result[0] += 0.0048679006;
            } else {
              result[0] += -0.043877617;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 116))) {
              result[0] += 0.014281779;
            } else {
              result[0] += 0.047004733;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 164))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 92))) {
              result[0] += -0.0021749143;
            } else {
              result[0] += -0.012218022;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
              result[0] += 0.0060172398;
            } else {
              result[0] += 0.033813428;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 228))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
              result[0] += -0.029434139;
            } else {
              result[0] += -0.014660637;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
              result[0] += -0.008835907;
            } else {
              result[0] += 0.0022141964;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 310))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 212))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 42))) {
              result[0] += 0.023493351;
            } else {
              result[0] += -0.0012283614;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
              result[0] += 0.002782393;
            } else {
              result[0] += 0.019489653;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 116))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 160))) {
              result[0] += 0.07486535;
            } else {
              result[0] += -0.009162361;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 168))) {
              result[0] += -0.0022064948;
            } else {
              result[0] += 0.018015308;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 128))) {
              result[0] += -0.01574024;
            } else {
              result[0] += -0.0039653988;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 140))) {
              result[0] += 0.020983376;
            } else {
              result[0] += 0.0030928531;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 238))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 244))) {
              result[0] += -0.018868484;
            } else {
              result[0] += -0.012392565;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 234))) {
              result[0] += 0.00805167;
            } else {
              result[0] += -0.011693968;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 86))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 96))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 84))) {
              result[0] += -0.038049206;
            } else {
              result[0] += -0.002771721;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 320))) {
              result[0] += 0.012239266;
            } else {
              result[0] += 0.03609361;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
            result[0] += 0.027658317;
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 136))) {
              result[0] += 0.051759947;
            } else {
              result[0] += 0.094818436;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 230))) {
              result[0] += 0.005940085;
            } else {
              result[0] += 0.017185425;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 300))) {
              result[0] += -0.02267663;
            } else {
              result[0] += -0.00748341;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 116))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 22))) {
              result[0] += 0.005678715;
            } else {
              result[0] += 0.02415133;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 76))) {
              result[0] += -0.026114738;
            } else {
              result[0] += 0.06968238;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 102))) {
    if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 144))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 18))) {
              result[0] += -0.042726804;
            } else {
              result[0] += -0.0135103455;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 162))) {
              result[0] += -0.007055785;
            } else {
              result[0] += 0.014698033;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 170))) {
              result[0] += -0.007809716;
            } else {
              result[0] += -0.025070379;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
              result[0] += -0;
            } else {
              result[0] += 0.02416715;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 242))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
              result[0] += -0.004747329;
            } else {
              result[0] += -0.030286456;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 38))) {
              result[0] += 0.004049407;
            } else {
              result[0] += -0.003883451;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 280))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 44))) {
              result[0] += 0.0048826905;
            } else {
              result[0] += 0.01918045;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 292))) {
              result[0] += -0.010770427;
            } else {
              result[0] += 0.010454954;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 126))) {
          result[0] += -0.04999471;
        } else {
          result[0] += -0.1126702;
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 6))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 8))) {
            result[0] += -0.0024832468;
          } else {
            result[0] += 0.034831993;
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 82))) {
              result[0] += 0.018345565;
            } else {
              result[0] += -0.03647999;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 72))) {
              result[0] += -0.025135571;
            } else {
              result[0] += -0;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
      if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 158))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 42))) {
              result[0] += -0.031185156;
            } else {
              result[0] += -0.011169832;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
              result[0] += -0.014554634;
            } else {
              result[0] += 0.05174926;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 134))) {
              result[0] += 0.033395614;
            } else {
              result[0] += -0.015813565;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 274))) {
              result[0] += 0.00033646842;
            } else {
              result[0] += 0.0078946445;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 114))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 154))) {
              result[0] += 0.008154803;
            } else {
              result[0] += -0.014786914;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
              result[0] += -0.06375181;
            } else {
              result[0] += 0.014291716;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 46))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
              result[0] += -0.0068405718;
            } else {
              result[0] += -0.034137234;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 288))) {
              result[0] += 0.027196486;
            } else {
              result[0] += -0.029254882;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 304))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 246))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 104))) {
              result[0] += -0.00011478314;
            } else {
              result[0] += -0.008755667;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 106))) {
              result[0] += 0.00041385577;
            } else {
              result[0] += 0.01698925;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 324))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
              result[0] += -0.008199277;
            } else {
              result[0] += -0.01730309;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
              result[0] += 0.0039300146;
            } else {
              result[0] += 0.015076749;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 218))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 372))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 232))) {
              result[0] += -0.0025358626;
            } else {
              result[0] += -0.014611224;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 16))) {
              result[0] += 0.021280123;
            } else {
              result[0] += -0.008725259;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
              result[0] += 0.04309545;
            } else {
              result[0] += 0.0142627945;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 34))) {
              result[0] += 0.024396539;
            } else {
              result[0] += 0.0048766937;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 170))) {
    if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 16))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 28))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 70))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 68))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
              result[0] += -0.024164472;
            } else {
              result[0] += 0.008937138;
            }
          } else {
            result[0] += 0.038625453;
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 76))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 98))) {
              result[0] += -0.04647706;
            } else {
              result[0] += -0.019371232;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 154))) {
              result[0] += -0.007284378;
            } else {
              result[0] += -0.041731447;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 14))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
              result[0] += -0.049801502;
            } else {
              result[0] += 0.007428868;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
              result[0] += 0.061795365;
            } else {
              result[0] += 0.022740616;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
              result[0] += 0.022688469;
            } else {
              result[0] += -0.016254207;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 170))) {
              result[0] += 0.012587378;
            } else {
              result[0] += -0.019465229;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 60))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 294))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 70))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += -0.016392779;
            } else {
              result[0] += 0.0076769763;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 132))) {
              result[0] += -0.0070873285;
            } else {
              result[0] += -0.017376283;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 180))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 58))) {
              result[0] += -0.008054349;
            } else {
              result[0] += 0.00049257686;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
              result[0] += 0.010894717;
            } else {
              result[0] += 0.0022375863;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 260))) {
              result[0] += -0.018788002;
            } else {
              result[0] += 0.009024569;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 266))) {
              result[0] += -0.0017262193;
            } else {
              result[0] += 0.015350415;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 172))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 194))) {
              result[0] += 0.01940038;
            } else {
              result[0] += 0.0023542785;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 112))) {
              result[0] += 0.003066004;
            } else {
              result[0] += -0.010361836;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 106))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 290))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 104))) {
              result[0] += -0.00023940679;
            } else {
              result[0] += -0.024299331;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 140))) {
              result[0] += 0.018921724;
            } else {
              result[0] += -0.012017216;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 20))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 304))) {
              result[0] += -0.009317097;
            } else {
              result[0] += 0.0070780194;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
              result[0] += 0.029872233;
            } else {
              result[0] += 0.0070561725;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
            result[0] += -0.012874908;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 140))) {
              result[0] += -0.03384575;
            } else {
              result[0] += -0.09957864;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 92))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
              result[0] += 0.007720273;
            } else {
              result[0] += 0.050167393;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 288))) {
              result[0] += 0.030057851;
            } else {
              result[0] += 0.012027963;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 2))) {
            result[0] += -0.03683686;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
              result[0] += -0.018220117;
            } else {
              result[0] += 0.0014780884;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 328))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 146))) {
              result[0] += 0.02194166;
            } else {
              result[0] += -0.00030332725;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 84))) {
              result[0] += -0.007878302;
            } else {
              result[0] += -0.016368117;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 90))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 104))) {
              result[0] += -0.015569873;
            } else {
              result[0] += -0.00039696257;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 2))) {
              result[0] += 0.022349734;
            } else {
              result[0] += 0.06912997;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
              result[0] += -0.0011338064;
            } else {
              result[0] += -0.015420935;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 218))) {
              result[0] += -0.0004335566;
            } else {
              result[0] += 0.012481102;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 226))) {
    if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
      if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 36))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 150))) {
              result[0] += -0.008768398;
            } else {
              result[0] += -0.031017033;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
              result[0] += -0.0019974862;
            } else {
              result[0] += -0.010033133;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 282))) {
              result[0] += 0.0018652402;
            } else {
              result[0] += 0.028417757;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
              result[0] += -0.02573219;
            } else {
              result[0] += 0.00014164243;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
              result[0] += -0.012296872;
            } else {
              result[0] += 0.018667158;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 52))) {
              result[0] += 0.0111978315;
            } else {
              result[0] += 0.00019159181;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 104))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
              result[0] += -0.018547757;
            } else {
              result[0] += 0.00039879596;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 20))) {
              result[0] += -0.025369791;
            } else {
              result[0] += -0.006152769;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
          result[0] += 0.0284749;
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 4))) {
            result[0] += -0.12034156;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 90))) {
              result[0] += 0.01555708;
            } else {
              result[0] += -0.04060086;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 28))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
              result[0] += -0.00899754;
            } else {
              result[0] += -0.041497704;
            }
          } else {
            result[0] += 0.012691744;
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 2))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 38))) {
              result[0] += -0.011100016;
            } else {
              result[0] += -0.065193616;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 34))) {
              result[0] += -0.026525358;
            } else {
              result[0] += -0.007171394;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 74))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
            result[0] += 0.017233951;
          } else {
            result[0] += 0.047289252;
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 292))) {
            result[0] += -0.020638349;
          } else {
            result[0] += 0.02203339;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 102))) {
          result[0] += -0.028595468;
        } else {
          result[0] += 0.035886366;
        }
      }
    } else {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
              result[0] += 0.023079382;
            } else {
              result[0] += -0.0015904735;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
              result[0] += 0.01960182;
            } else {
              result[0] += 0.004229198;
            }
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 110))) {
            result[0] += 0.042509545;
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 78))) {
              result[0] += 0.01140577;
            } else {
              result[0] += 0.03407;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 134))) {
              result[0] += -0.002405419;
            } else {
              result[0] += -0.010636522;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 60))) {
              result[0] += 0.012451133;
            } else {
              result[0] += -0.0039769937;
            }
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
              result[0] += -0.01115589;
            } else {
              result[0] += 0.01320644;
            }
          } else {
            result[0] += -0.013309436;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 90))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
      if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 24))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 46))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 18))) {
              result[0] += -0.054960348;
            } else {
              result[0] += -0.008308151;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 116))) {
              result[0] += 0.040893547;
            } else {
              result[0] += -0.029395109;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 108))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 14))) {
              result[0] += -0.011575009;
            } else {
              result[0] += 0.0045230165;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
              result[0] += -0.01649066;
            } else {
              result[0] += 0.0117938975;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 28))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 92))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 44))) {
              result[0] += -0.017786995;
            } else {
              result[0] += 0.0030950578;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 38))) {
              result[0] += -0.044471394;
            } else {
              result[0] += 0.022061352;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 52))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 114))) {
              result[0] += -0.014604069;
            } else {
              result[0] += 0.016241714;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 42))) {
              result[0] += 0.018912865;
            } else {
              result[0] += 0.0019412668;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 306))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 64))) {
              result[0] += -0.006083592;
            } else {
              result[0] += 0.046224184;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += 0.015739193;
            } else {
              result[0] += -0.022769345;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
            result[0] += 0.03625956;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 212))) {
              result[0] += 0.00036286406;
            } else {
              result[0] += 0.009678096;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 240))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 230))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
              result[0] += -0.010175624;
            } else {
              result[0] += 0.0050961208;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 46))) {
              result[0] += -0.0105675245;
            } else {
              result[0] += -0.026643395;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 226))) {
            result[0] += -0;
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 86))) {
              result[0] += 0.023571983;
            } else {
              result[0] += -0;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 222))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 106))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 66))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 54))) {
              result[0] += -0.07242959;
            } else {
              result[0] += -0.00901923;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 190))) {
              result[0] += 0.0003989732;
            } else {
              result[0] += -0.014672895;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 56))) {
              result[0] += -0.0052175657;
            } else {
              result[0] += 0.038002204;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 86))) {
              result[0] += -0.030148897;
            } else {
              result[0] += -0.006241556;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
              result[0] += -5.1738414e-05;
            } else {
              result[0] += 0.0063861213;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
              result[0] += 0.022810614;
            } else {
              result[0] += -0.016576035;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 234))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.0050715203;
            } else {
              result[0] += 0.005531682;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 238))) {
              result[0] += -0.01301794;
            } else {
              result[0] += 0.00030336596;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 180))) {
              result[0] += -0.0094006555;
            } else {
              result[0] += 0.02063076;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
              result[0] += 0.010461995;
            } else {
              result[0] += 0.0025512462;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 62))) {
            result[0] += 0.010800346;
          } else {
            result[0] += 0.052793957;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 16))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
              result[0] += -0.02969619;
            } else {
              result[0] += -0.01354564;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 126))) {
              result[0] += -0.0019324312;
            } else {
              result[0] += 0.018997952;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 228))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
              result[0] += 0.01697753;
            } else {
              result[0] += -0.0008568109;
            }
          } else {
            result[0] += -0.0515616;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 106))) {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 30))) {
      if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 18))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 86))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 40))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 14))) {
              result[0] += -0.036433112;
            } else {
              result[0] += -0.011925978;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 56))) {
              result[0] += 0.03285799;
            } else {
              result[0] += 0.0010874722;
            }
          }
        } else {
          result[0] += -0.060846817;
        }
      } else {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 92))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 34))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += -0.012450369;
            } else {
              result[0] += -0.04985917;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 24))) {
              result[0] += 0.0078088613;
            } else {
              result[0] += -0.00958295;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 168))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 24))) {
              result[0] += -0.043195866;
            } else {
              result[0] += -0.014012662;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 26))) {
              result[0] += -0.019311419;
            } else {
              result[0] += 0.007864608;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 152))) {
              result[0] += -0.012414033;
            } else {
              result[0] += 0.029668832;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
              result[0] += -0.0030567853;
            } else {
              result[0] += 0.01868884;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 254))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
              result[0] += 0.026466904;
            } else {
              result[0] += -0.00078509795;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 256))) {
              result[0] += -0.017463585;
            } else {
              result[0] += -0.004959351;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 90))) {
              result[0] += 0.054318376;
            } else {
              result[0] += 0.016744314;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
              result[0] += -0.008082875;
            } else {
              result[0] += 0.004647791;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 136))) {
              result[0] += -0.019794006;
            } else {
              result[0] += 0.037679102;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += 0.044813335;
            } else {
              result[0] += -0.0018291153;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 68))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 76))) {
              result[0] += -0.011341448;
            } else {
              result[0] += -0.060769796;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 306))) {
              result[0] += -0.00012510354;
            } else {
              result[0] += 0.019792138;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 88))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 110))) {
              result[0] += -0.005990359;
            } else {
              result[0] += 0.017361904;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
              result[0] += 0.008660458;
            } else {
              result[0] += 0.0014748373;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 220))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 272))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 118))) {
              result[0] += -0.0027236396;
            } else {
              result[0] += -0.029521126;
            }
          } else {
            result[0] += -0;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += -0.058680933;
            } else {
              result[0] += 0.0022811424;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 206))) {
              result[0] += -0.017125372;
            } else {
              result[0] += 0.008216227;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
        result[0] += -0.01995611;
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 30))) {
          result[0] += 0.021725012;
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 220))) {
            result[0] += 0.050011344;
          } else {
            result[0] += 0.014028007;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 56))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 4))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 104))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 44))) {
              result[0] += 0.021376295;
            } else {
              result[0] += 0.0036938381;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 274))) {
              result[0] += -0.0331207;
            } else {
              result[0] += -0.005488401;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 8))) {
              result[0] += 0.015083553;
            } else {
              result[0] += -0.016786192;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 30))) {
              result[0] += -0.037689086;
            } else {
              result[0] += -0.019093947;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 130))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 8))) {
            result[0] += 0.059979796;
          } else {
            result[0] += 0.0069023515;
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 290))) {
            result[0] += -0.024388736;
          } else {
            result[0] += -0;
          }
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 272))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 52))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 64))) {
              result[0] += 0.030230714;
            } else {
              result[0] += -0.002469063;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.00685675;
            } else {
              result[0] += -0.019919405;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 40))) {
              result[0] += 0.007914934;
            } else {
              result[0] += -0.0072328295;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 168))) {
              result[0] += -0.023456104;
            } else {
              result[0] += -0.007590213;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 292))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 278))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 268))) {
              result[0] += -0.00084484427;
            } else {
              result[0] += 0.008248358;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 98))) {
              result[0] += 0.008069568;
            } else {
              result[0] += -0.01922637;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 58))) {
            result[0] += -0.0034377978;
          } else {
            result[0] += 0.02219229;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 114))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 120))) {
              result[0] += -0.0008113146;
            } else {
              result[0] += -0.011595509;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 30))) {
              result[0] += -0.00054977357;
            } else {
              result[0] += 0.0060809394;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
              result[0] += -0.013630775;
            } else {
              result[0] += 0.029257402;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
              result[0] += 0.01473201;
            } else {
              result[0] += 0.004033634;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 326))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += 0.0005843948;
            } else {
              result[0] += -0.015655208;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 368))) {
              result[0] += -0.007192479;
            } else {
              result[0] += 0.006543958;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 346))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 10))) {
              result[0] += 0.01452767;
            } else {
              result[0] += 0.0055364254;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 372))) {
              result[0] += 0.026855484;
            } else {
              result[0] += -0.005540358;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 88))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 92))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 60))) {
            result[0] += 0.04485644;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
              result[0] += -0.042145204;
            } else {
              result[0] += 7.136203e-05;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 32))) {
              result[0] += -0.005530101;
            } else {
              result[0] += -0.041934248;
            }
          } else {
            result[0] += -0.07572525;
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 246))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 282))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
              result[0] += 0.025418038;
            } else {
              result[0] += -0.0084718885;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 214))) {
              result[0] += -0.019256549;
            } else {
              result[0] += 0.00998631;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 98))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 130))) {
              result[0] += -0.054605898;
            } else {
              result[0] += -0.011759637;
            }
          } else {
            result[0] += -0.046184976;
          }
        }
      }
    }
  }
  if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 176))) {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 30))) {
      if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 32))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 36))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
              result[0] += -0.022035845;
            } else {
              result[0] += 0.046122007;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
              result[0] += 0.0077522853;
            } else {
              result[0] += -0.02736899;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 128))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 120))) {
              result[0] += -0.0018629817;
            } else {
              result[0] += 0.02747905;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += 0.02887457;
            } else {
              result[0] += -0.013981772;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 108))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
              result[0] += -0.01683332;
            } else {
              result[0] += -0.0050804014;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 52))) {
              result[0] += 0.03178656;
            } else {
              result[0] += -0.0057087517;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 90))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 152))) {
              result[0] += -0.014723192;
            } else {
              result[0] += -0.04805673;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 144))) {
              result[0] += -0.017328734;
            } else {
              result[0] += -0.03817048;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 42))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 28))) {
              result[0] += -0.0028061648;
            } else {
              result[0] += 0.002789692;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
              result[0] += -0.025860507;
            } else {
              result[0] += -0.0057830475;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 330))) {
              result[0] += 0.006883178;
            } else {
              result[0] += -0.010597587;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 66))) {
              result[0] += -0.03156317;
            } else {
              result[0] += -0.00012491019;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 36))) {
          result[0] += -0.049023475;
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
              result[0] += -0.015445833;
            } else {
              result[0] += -0.038803425;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 164))) {
              result[0] += -0.0042772954;
            } else {
              result[0] += -0.02979202;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 4))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 18))) {
          result[0] += -0.105805;
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 156))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 218))) {
              result[0] += -0.013860464;
            } else {
              result[0] += -0.047610518;
            }
          } else {
            result[0] += 0.015889838;
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 296))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 170))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 150))) {
              result[0] += 0.0039272998;
            } else {
              result[0] += 0.04100207;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
              result[0] += 0.066427186;
            } else {
              result[0] += 0.019468648;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 354))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += -0.013523097;
            } else {
              result[0] += -0.035645615;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 302))) {
              result[0] += -0.017095027;
            } else {
              result[0] += 0.008768201;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 84))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 196))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 88))) {
              result[0] += -0.0047009015;
            } else {
              result[0] += 0.0095943585;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
              result[0] += -0.024147784;
            } else {
              result[0] += -0.007935442;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 100))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 360))) {
              result[0] += -0.025299666;
            } else {
              result[0] += -0.015776524;
            }
          } else {
            result[0] += -0.00817149;
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 90))) {
            result[0] += -0.013373877;
          } else {
            result[0] += 0.06051935;
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 112))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 54))) {
              result[0] += 0.013844582;
            } else {
              result[0] += -0.018000482;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 116))) {
              result[0] += 0.025009016;
            } else {
              result[0] += 0.0011969743;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 6))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 46))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 96))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 86))) {
          result[0] += -0.005377132;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
              result[0] += 0.033936527;
            } else {
              result[0] += 0.06770065;
            }
          } else {
            result[0] += -0;
          }
        }
      } else {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 34))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 28))) {
              result[0] += -0.016085489;
            } else {
              result[0] += 0.012783072;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 26))) {
              result[0] += -0.038858794;
            } else {
              result[0] += -0.004968185;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 36))) {
            result[0] += 0.04801505;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
              result[0] += 0.01772366;
            } else {
              result[0] += -0.026647871;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 84))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
              result[0] += -0.02566973;
            } else {
              result[0] += 0.028348282;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 120))) {
              result[0] += -0.043836653;
            } else {
              result[0] += -0.012338279;
            }
          }
        } else {
          result[0] += -0.10324136;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 36))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 2))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 12))) {
              result[0] += -0.033641063;
            } else {
              result[0] += 0.00022004887;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 116))) {
              result[0] += -0.044643078;
            } else {
              result[0] += -0.008762143;
            }
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
              result[0] += 0.0021203114;
            } else {
              result[0] += -0.017186115;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 88))) {
              result[0] += -0.019821104;
            } else {
              result[0] += 0.0021318241;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 184))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 222))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
              result[0] += 0.0009353072;
            } else {
              result[0] += -0.0056581837;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 170))) {
              result[0] += -0.0009079633;
            } else {
              result[0] += 0.008418952;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 108))) {
              result[0] += -0;
            } else {
              result[0] += -0.03857514;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 26))) {
              result[0] += 0.0028833076;
            } else {
              result[0] += 0.030077396;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 86))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 66))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 44))) {
              result[0] += 0.0097111445;
            } else {
              result[0] += -0.004844829;
            }
          } else {
            result[0] += 0.022152675;
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 58))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 36))) {
              result[0] += 0.012857464;
            } else {
              result[0] += 0.04939564;
            }
          } else {
            result[0] += 0.0038712837;
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 194))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 96))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 142))) {
              result[0] += 0.022889886;
            } else {
              result[0] += -0.025423696;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 194))) {
              result[0] += 0.0012433341;
            } else {
              result[0] += -0.027690545;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 42))) {
            result[0] += 0.019493537;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 248))) {
              result[0] += 0.04888856;
            } else {
              result[0] += 0.0047187754;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 234))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
              result[0] += -0.038507808;
            } else {
              result[0] += -0.0059399325;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 190))) {
              result[0] += -0.00081904867;
            } else {
              result[0] += 0.023750668;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 62))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 180))) {
              result[0] += 0.007010121;
            } else {
              result[0] += 0.00020330278;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 342))) {
              result[0] += 0.022976872;
            } else {
              result[0] += -0.008908423;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 16))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 20))) {
        result[0] += 0.04029838;
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 0))) {
            result[0] += -0.0076732375;
          } else {
            result[0] += -0.044110678;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 140))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 138))) {
              result[0] += 0.013151512;
            } else {
              result[0] += 0.04138301;
            }
          } else {
            result[0] += -0.031063614;
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 38))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 244))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 212))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 128))) {
              result[0] += -0.01187767;
            } else {
              result[0] += -0.024798945;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += -0.014814547;
            } else {
              result[0] += -0.0693584;
            }
          }
        } else {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
            result[0] += -0.019700082;
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 12))) {
              result[0] += 0.020823775;
            } else {
              result[0] += -0.0007175183;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 28))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
              result[0] += -0.04782397;
            } else {
              result[0] += -0.00023042485;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 18))) {
              result[0] += 0.011482454;
            } else {
              result[0] += -0.025828404;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 30))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 68))) {
              result[0] += 0.07567041;
            } else {
              result[0] += 0.03777515;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 36))) {
              result[0] += 0.019245345;
            } else {
              result[0] += -0.007408004;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 298))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
              result[0] += -0.0023852098;
            } else {
              result[0] += -0.01571614;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
              result[0] += -0.0008245474;
            } else {
              result[0] += -0.01959662;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 52))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 160))) {
              result[0] += 0.01735154;
            } else {
              result[0] += -0.01864585;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 104))) {
              result[0] += -0.0019429872;
            } else {
              result[0] += 0.001837825;
            }
          }
        }
      } else {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 86))) {
          result[0] += 0.013189456;
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
            result[0] += 0.007143569;
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 44))) {
              result[0] += 0.025064811;
            } else {
              result[0] += 0.0470048;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 138))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 2))) {
          result[0] += -0.028255967;
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 330))) {
              result[0] += 0.0038780514;
            } else {
              result[0] += 0.027806832;
            }
          } else {
            result[0] += -0.030956248;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 144))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 98))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 170))) {
              result[0] += 0.0049754553;
            } else {
              result[0] += 0.07370368;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 54))) {
              result[0] += 0.0024109604;
            } else {
              result[0] += -0.021205995;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 232))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
              result[0] += 0.006072202;
            } else {
              result[0] += 0.021960644;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 60))) {
              result[0] += -0.016412217;
            } else {
              result[0] += 0.02891737;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 176))) {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 32))) {
              result[0] += -0.013008319;
            } else {
              result[0] += 0.014044277;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 8))) {
              result[0] += -0.04008707;
            } else {
              result[0] += 0.011214378;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 52))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 8))) {
              result[0] += -0.021072047;
            } else {
              result[0] += -0.032819312;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 16))) {
              result[0] += -0.014122496;
            } else {
              result[0] += -0.0027466726;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
              result[0] += 0.003206402;
            } else {
              result[0] += 0.023088232;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
              result[0] += -0.029293472;
            } else {
              result[0] += -0.00044284094;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 88))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 84))) {
              result[0] += -0.037337154;
            } else {
              result[0] += -0.054280948;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
              result[0] += 0.000117012045;
            } else {
              result[0] += -0.009735559;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 152))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 194))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 12))) {
              result[0] += -0.024764068;
            } else {
              result[0] += 0.0052268724;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 64))) {
              result[0] += 0.014035463;
            } else {
              result[0] += 0.042725824;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 78))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 118))) {
              result[0] += -0.016555734;
            } else {
              result[0] += 0.0024362335;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 132))) {
              result[0] += 0.013627543;
            } else {
              result[0] += 0.0015473928;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 168))) {
          result[0] += 0.04074187;
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 14))) {
            result[0] += 0.029437272;
          } else {
            result[0] += 0.0063492656;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 36))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 28))) {
              result[0] += -0.007086392;
            } else {
              result[0] += -0.028627744;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 70))) {
              result[0] += -0.01253838;
            } else {
              result[0] += 0.020658642;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 100))) {
            result[0] += -0.015626702;
          } else {
            result[0] += -0.07547299;
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
          result[0] += -0.098590024;
        } else {
          result[0] += -0.016383274;
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 156))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 262))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 238))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 148))) {
              result[0] += -0.005975528;
            } else {
              result[0] += -0.017585494;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 134))) {
              result[0] += -0.06808423;
            } else {
              result[0] += -0.022014553;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 292))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 128))) {
              result[0] += -0.010686524;
            } else {
              result[0] += 0.011010774;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 284))) {
              result[0] += 0.03166331;
            } else {
              result[0] += 0.0053174673;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 82))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 258))) {
            result[0] += -0.049204163;
          } else {
            result[0] += -0.0052781263;
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 108))) {
            result[0] += 0.007684131;
          } else {
            result[0] += -0.01196891;
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 250))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 30))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 168))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 16))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 120))) {
              result[0] += -0;
            } else {
              result[0] += 0.025176907;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 158))) {
              result[0] += -0.009614422;
            } else {
              result[0] += -0.025206959;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 216))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 82))) {
              result[0] += -0.0041216165;
            } else {
              result[0] += 0.042651124;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 84))) {
              result[0] += -0.038773373;
            } else {
              result[0] += 0.023100061;
            }
          }
        }
      } else {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.006539114;
            } else {
              result[0] += -0.062450953;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += -0.004460583;
            } else {
              result[0] += 0.0019909479;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
              result[0] += -0.00021329732;
            } else {
              result[0] += 0.011201916;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
              result[0] += 0.01424145;
            } else {
              result[0] += 0.0014522666;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 254))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 50))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
              result[0] += 0.033733904;
            } else {
              result[0] += -0.0006381403;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 28))) {
              result[0] += -0.015036389;
            } else {
              result[0] += 0.0023278003;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 2))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 280))) {
              result[0] += -0.020510865;
            } else {
              result[0] += 0.02703706;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 52))) {
              result[0] += 0.022050345;
            } else {
              result[0] += -0.0019060083;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 118))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 314))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 300))) {
              result[0] += -0.019168613;
            } else {
              result[0] += 0.006976162;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 264))) {
              result[0] += -0.015453433;
            } else {
              result[0] += 0.0013623396;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 60))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 206))) {
              result[0] += 0.004762128;
            } else {
              result[0] += -0.006663674;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 66))) {
              result[0] += 0.025200857;
            } else {
              result[0] += -0.005030225;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 54))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 44))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 42))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 2))) {
              result[0] += -0.012792389;
            } else {
              result[0] += 0.017205698;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += -0.042184174;
            } else {
              result[0] += 0.004857573;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 66))) {
              result[0] += -0.028872743;
            } else {
              result[0] += -0.006002301;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 12))) {
              result[0] += -0.0013448104;
            } else {
              result[0] += -0.04370745;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 214))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 302))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
              result[0] += 0.0069149467;
            } else {
              result[0] += -0.0046409685;
            }
          } else {
            result[0] += 0.02378996;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 62))) {
              result[0] += 0.012926725;
            } else {
              result[0] += -0.008747402;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 50))) {
              result[0] += -0.046444967;
            } else {
              result[0] += -0.018540045;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 146))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 134))) {
          result[0] += -0.09067112;
        } else {
          result[0] += -0.010558448;
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
          result[0] += -0.023714645;
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 194))) {
            result[0] += -0.010494131;
          } else {
            result[0] += 0.01534995;
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 308))) {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 110))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
              result[0] += -0.03064735;
            } else {
              result[0] += 0.0034591525;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 96))) {
              result[0] += 0.026732335;
            } else {
              result[0] += 0.008400806;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
              result[0] += -0.005901861;
            } else {
              result[0] += 0.009023737;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += -0.012533008;
            } else {
              result[0] += -0.0020645715;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 22))) {
              result[0] += -0.013980149;
            } else {
              result[0] += 0.0019120906;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 54))) {
              result[0] += -0.0057765585;
            } else {
              result[0] += 0.017128821;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
              result[0] += -0.018316587;
            } else {
              result[0] += -0.005805789;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
              result[0] += 0.03265365;
            } else {
              result[0] += -0.0010156564;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 240))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 122))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 184))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 132))) {
              result[0] += 0.014779656;
            } else {
              result[0] += -0.020928297;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 206))) {
              result[0] += 0.07677117;
            } else {
              result[0] += -0.004149875;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 254))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 108))) {
              result[0] += 0.0061775236;
            } else {
              result[0] += -0.002984956;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 192))) {
              result[0] += -0.008194422;
            } else {
              result[0] += -0.05728069;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 192))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 280))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 0))) {
              result[0] += 0.022544114;
            } else {
              result[0] += 0.0022025593;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
              result[0] += -0.004058608;
            } else {
              result[0] += 0.014417312;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 318))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 338))) {
              result[0] += 0.0165982;
            } else {
              result[0] += 0.038578045;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
              result[0] += -0.015475737;
            } else {
              result[0] += 0.01142684;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 20))) {
        result[0] += -0.0928144;
      } else {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 180))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 46))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
              result[0] += -0.020081904;
            } else {
              result[0] += 0.002259192;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
              result[0] += -0.07451444;
            } else {
              result[0] += -0.0156236235;
            }
          }
        } else {
          result[0] += -0.07355082;
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 108))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 38))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
              result[0] += -0.037877556;
            } else {
              result[0] += 0.0013853526;
            }
          } else {
            result[0] += -0.050581098;
          }
        } else {
          result[0] += -0;
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 178))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 176))) {
              result[0] += -0.0043627243;
            } else {
              result[0] += -0.020921404;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 24))) {
              result[0] += 0.01141603;
            } else {
              result[0] += -0.0036946;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 228))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 94))) {
              result[0] += -0.027430529;
            } else {
              result[0] += 0.00075300987;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 74))) {
              result[0] += -0.05395777;
            } else {
              result[0] += -0.013114258;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 226))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 270))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 266))) {
              result[0] += -0.0013336329;
            } else {
              result[0] += -0.012101962;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 96))) {
              result[0] += -0.012478279;
            } else {
              result[0] += -0.042821463;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 108))) {
              result[0] += -0.00018370214;
            } else {
              result[0] += 0.018782655;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 90))) {
              result[0] += 0.0019190798;
            } else {
              result[0] += -0.0032492876;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 114))) {
          result[0] += -0.041382577;
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
            result[0] += 0.019699065;
          } else {
            result[0] += -0.023365228;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 242))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 152))) {
              result[0] += -0.014805789;
            } else {
              result[0] += -5.5329445e-05;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 276))) {
              result[0] += 0.031025484;
            } else {
              result[0] += 0.00553753;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 282))) {
            result[0] += 0.03033769;
          } else {
            result[0] += 0.053280126;
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += 0.0059688077;
            } else {
              result[0] += -0.027727133;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 314))) {
              result[0] += 0.009310781;
            } else {
              result[0] += -0.008495542;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 40))) {
            result[0] += 0.038052257;
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 12))) {
              result[0] += 0.012323047;
            } else {
              result[0] += -0.023586523;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 144))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 130))) {
              result[0] += 0.0018252156;
            } else {
              result[0] += -0.00692114;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 312))) {
              result[0] += 0.032022245;
            } else {
              result[0] += -0.006053284;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 338))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 132))) {
              result[0] += -0.019810393;
            } else {
              result[0] += -0.0073889843;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 36))) {
              result[0] += 0.016521052;
            } else {
              result[0] += 0.0061617037;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 76))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 244))) {
            result[0] += 0.012791178;
          } else {
            result[0] += -0.020304035;
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 258))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
              result[0] += 0.009074631;
            } else {
              result[0] += -0.0016534543;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
              result[0] += 0.027765274;
            } else {
              result[0] += -0.0001559796;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 234))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 258))) {
          result[0] += -0.009072641;
        } else {
          result[0] += 0.008848451;
        }
      } else {
        result[0] += -0.012117184;
      }
    }
  }
  if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 6))) {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 148))) {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
            result[0] += 0.009714896;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 18))) {
              result[0] += -0.035071608;
            } else {
              result[0] += -0.015023454;
            }
          }
        } else {
          result[0] += -0.073392645;
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 90))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 132))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 38))) {
              result[0] += -0.0044015315;
            } else {
              result[0] += 0.012927837;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 4))) {
              result[0] += -0.0023782412;
            } else {
              result[0] += -0.021760104;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 42))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 112))) {
              result[0] += -0.0147892255;
            } else {
              result[0] += 0.0073062084;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
              result[0] += -0.026250172;
            } else {
              result[0] += 0.0071073617;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 0))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 162))) {
          result[0] += 0.040685426;
        } else {
          result[0] += 0.0120480135;
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 308))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 18))) {
            result[0] += -0.06254916;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 98))) {
              result[0] += -0.0067300186;
            } else {
              result[0] += -0.0331382;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
            result[0] += -0.019220524;
          } else {
            result[0] += 0.009100321;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 120))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 206))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
              result[0] += 0.009733505;
            } else {
              result[0] += -0.004346859;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
              result[0] += 8.9312605e-05;
            } else {
              result[0] += -0.003453482;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 24))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 280))) {
              result[0] += -0.0062282663;
            } else {
              result[0] += 0.0049122893;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += -0.002204797;
            } else {
              result[0] += 0.0064123496;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 46))) {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 44))) {
              result[0] += 0.06116004;
            } else {
              result[0] += 0.010365079;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 146))) {
              result[0] += -0.018945072;
            } else {
              result[0] += -0.010433878;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 118))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 168))) {
              result[0] += 0.01741377;
            } else {
              result[0] += 0.07594368;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 170))) {
              result[0] += -0.0018027859;
            } else {
              result[0] += 0.016592477;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
        result[0] += 0.043439865;
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 82))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 58))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
              result[0] += -0.036944382;
            } else {
              result[0] += -0.012312688;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.043840874;
            } else {
              result[0] += -0.014361912;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.004802558;
            } else {
              result[0] += 0.007698638;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 170))) {
              result[0] += -0.017537218;
            } else {
              result[0] += -0.009052196;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 10))) {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 76))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 18))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 16))) {
              result[0] += -0.011083994;
            } else {
              result[0] += -0.033235386;
            }
          } else {
            result[0] += 0.03099644;
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 2))) {
            result[0] += -0.020493729;
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 102))) {
              result[0] += 0.03927904;
            } else {
              result[0] += -0.0013705323;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 172))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 80))) {
              result[0] += 0.06984255;
            } else {
              result[0] += 0.024543915;
            }
          } else {
            result[0] += 0.008717417;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
            result[0] += -0.034200173;
          } else {
            result[0] += 0.020928657;
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 40))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 34))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 30))) {
              result[0] += -0.018539859;
            } else {
              result[0] += 0.006696008;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
              result[0] += 0.0006399334;
            } else {
              result[0] += 0.023955312;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 46))) {
            result[0] += 0.0061688256;
          } else {
            result[0] += 0.044047784;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 134))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 102))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 156))) {
              result[0] += -0.016056351;
            } else {
              result[0] += 0.0054231877;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 76))) {
              result[0] += -0.0025266295;
            } else {
              result[0] += 0.018242473;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 270))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 104))) {
              result[0] += -0.017715478;
            } else {
              result[0] += -0.04547394;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 152))) {
              result[0] += 0.013087104;
            } else {
              result[0] += -0.029149592;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 264))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 52))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 188))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 22))) {
              result[0] += 0.0041657714;
            } else {
              result[0] += -0.022834495;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 272))) {
              result[0] += 0.016674465;
            } else {
              result[0] += 0.032325614;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 176))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 154))) {
              result[0] += -0.017698428;
            } else {
              result[0] += 0.03289035;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 240))) {
              result[0] += 0.0020410677;
            } else {
              result[0] += -0.020820608;
            }
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 74))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 70))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 50))) {
              result[0] += -4.355009e-05;
            } else {
              result[0] += -0.032748487;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 78))) {
              result[0] += 0.035664223;
            } else {
              result[0] += -0.0039311326;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += -0.0035012837;
            } else {
              result[0] += 0.02525419;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
              result[0] += -0.0004375557;
            } else {
              result[0] += -0.032447618;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 68))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 284))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 76))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 202))) {
              result[0] += 0.06691403;
            } else {
              result[0] += 0.026116902;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 228))) {
              result[0] += -0.0024726556;
            } else {
              result[0] += 0.0035805923;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 112))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 170))) {
              result[0] += 0.008209762;
            } else {
              result[0] += 0.046135128;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 142))) {
              result[0] += -0.0095024295;
            } else {
              result[0] += 0.0053031766;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 156))) {
          result[0] += 0.02656824;
        } else {
          result[0] += -0.00404437;
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 142))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 40))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 46))) {
              result[0] += 0.06587588;
            } else {
              result[0] += 0.0105197765;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 70))) {
              result[0] += 0.0041911616;
            } else {
              result[0] += -0.012468158;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
            result[0] += 0.009541268;
          } else {
            result[0] += 0.058547605;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 152))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 290))) {
              result[0] += -0.021359572;
            } else {
              result[0] += -0.004128001;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 162))) {
              result[0] += 0.011844903;
            } else {
              result[0] += 0.023288697;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 316))) {
              result[0] += -7.127906e-05;
            } else {
              result[0] += 0.0055441954;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 44))) {
              result[0] += -0.033032846;
            } else {
              result[0] += 0.008700618;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 100))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 28))) {
              result[0] += -0.020282162;
            } else {
              result[0] += -0;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 252))) {
              result[0] += 0.0372028;
            } else {
              result[0] += 0.020493207;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 114))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 192))) {
              result[0] += 0.0023391566;
            } else {
              result[0] += -0.012876572;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
              result[0] += 0.028000802;
            } else {
              result[0] += 0.008097276;
            }
          }
        }
      } else {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 12))) {
          result[0] += -0.030859623;
        } else {
          result[0] += 0.059427258;
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 216))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 296))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 12))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 158))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
              result[0] += 0.023903606;
            } else {
              result[0] += -0.01342114;
            }
          } else {
            result[0] += -0.03418419;
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 52))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 150))) {
              result[0] += -0.014690627;
            } else {
              result[0] += -0.046283614;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 34))) {
              result[0] += -0.018795181;
            } else {
              result[0] += -0.005358623;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 162))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 300))) {
              result[0] += -0.013268234;
            } else {
              result[0] += 0.011121328;
            }
          } else {
            result[0] += -0.0279394;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 176))) {
            result[0] += 0.03626206;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 326))) {
              result[0] += -0.0057000555;
            } else {
              result[0] += 0.0147212865;
            }
          }
        }
      }
    } else {
      result[0] += -0.06643397;
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 6))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 48))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 126))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 16))) {
              result[0] += -0.0010974932;
            } else {
              result[0] += 0.020823766;
            }
          } else {
            result[0] += 0.0449019;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 34))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
              result[0] += -0.008742712;
            } else {
              result[0] += 0.016725272;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 36))) {
              result[0] += -0.026789738;
            } else {
              result[0] += 0.005821276;
            }
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
          result[0] += -0.033222493;
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 40))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += -0.0058179474;
            } else {
              result[0] += 0.04347635;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 112))) {
              result[0] += -0.015091679;
            } else {
              result[0] += 0.0024027491;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 232))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 32))) {
            result[0] += 0.0399952;
          } else {
            result[0] += -0.00027889368;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 62))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
              result[0] += -0.01870556;
            } else {
              result[0] += 0.0057035587;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
              result[0] += 0.02278183;
            } else {
              result[0] += -0.008421278;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 20))) {
          result[0] += -0.042912465;
        } else {
          result[0] += -0;
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 242))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 160))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 194))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 110))) {
              result[0] += 0.003376493;
            } else {
              result[0] += 0.011122108;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
              result[0] += 0.03045251;
            } else {
              result[0] += 0.012394785;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
            result[0] += -0.027083784;
          } else {
            result[0] += 0.0126619935;
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
          result[0] += 0.008825673;
        } else {
          result[0] += 0.044847894;
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 144))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 10))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 96))) {
              result[0] += 0.023729367;
            } else {
              result[0] += -0.016748834;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 64))) {
              result[0] += -0.020931631;
            } else {
              result[0] += -0.05402078;
            }
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 30))) {
            result[0] += 0.020663498;
          } else {
            result[0] += 0.00616918;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 20))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 122))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 12))) {
              result[0] += -0.017938469;
            } else {
              result[0] += 0.03910426;
            }
          } else {
            result[0] += 0.0053874785;
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 14))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.022196533;
            } else {
              result[0] += -0.00451347;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
              result[0] += 0.00037458728;
            } else {
              result[0] += 0.01688151;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 254))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 22))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 90))) {
              result[0] += 0.027586067;
            } else {
              result[0] += -0.029394219;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 212))) {
              result[0] += -0.008463705;
            } else {
              result[0] += -0.028911496;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 194))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 154))) {
              result[0] += -0.0010301826;
            } else {
              result[0] += 0.0035315962;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 230))) {
              result[0] += -0.0011929054;
            } else {
              result[0] += -0.00760438;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 62))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 110))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.011337014;
            } else {
              result[0] += 0.0015774952;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 86))) {
              result[0] += -0.0043284963;
            } else {
              result[0] += 0.0071344683;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 256))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 206))) {
              result[0] += 0.0046818857;
            } else {
              result[0] += 0.01623973;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 332))) {
              result[0] += -0.025880137;
            } else {
              result[0] += 0.00024090149;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 46))) {
        if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 44))) {
            result[0] += 0.062326457;
          } else {
            result[0] += -0.012353135;
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 154))) {
            result[0] += -0.01557938;
          } else {
            result[0] += -0.007663093;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 114))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 156))) {
            result[0] += 0.07925903;
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 160))) {
              result[0] += -0.022038413;
            } else {
              result[0] += 0.010596351;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 118))) {
              result[0] += 0.009866733;
            } else {
              result[0] += 0.04229749;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 78))) {
              result[0] += -0.027659763;
            } else {
              result[0] += -0.0035324506;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 168))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 130))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 84))) {
            result[0] += -0.01782277;
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 212))) {
              result[0] += -0;
            } else {
              result[0] += -0.01930575;
            }
          }
        } else {
          result[0] += 0.0040967353;
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 402))) {
          result[0] += -0.021184763;
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 140))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.0055727926;
            } else {
              result[0] += 0.009987557;
            }
          } else {
            result[0] += -0.016817084;
          }
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 128))) {
              result[0] += -0.011004246;
            } else {
              result[0] += 0.0059432825;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 224))) {
              result[0] += -0.0151095195;
            } else {
              result[0] += -0.0033481915;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 92))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
              result[0] += 0.0145778125;
            } else {
              result[0] += -0.0010819718;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 182))) {
              result[0] += -0.01369258;
            } else {
              result[0] += 0.010227516;
            }
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 238))) {
          result[0] += -0.009665318;
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 180))) {
            result[0] += 0.026355088;
          } else {
            result[0] += -0.0077078748;
          }
        }
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 230))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 268))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 52))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 250))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
              result[0] += -0.0125539;
            } else {
              result[0] += 0.014108579;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
              result[0] += 0.021760508;
            } else {
              result[0] += -0.019894226;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 268))) {
            result[0] += -0.029441832;
          } else {
            result[0] += -0.010534534;
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 112))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 68))) {
              result[0] += -0.031425703;
            } else {
              result[0] += -0.019633126;
            }
          } else {
            result[0] += -0.01066619;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 210))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
              result[0] += -0.0006799324;
            } else {
              result[0] += -0.027934993;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
              result[0] += -0.015189849;
            } else {
              result[0] += -0.0026831003;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 236))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 166))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
              result[0] += -1.1786798e-05;
            } else {
              result[0] += 0.02334137;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 290))) {
              result[0] += 0.0047730226;
            } else {
              result[0] += 0.01909187;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 284))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 246))) {
              result[0] += 0.024219202;
            } else {
              result[0] += 0.04463927;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 28))) {
              result[0] += -0;
            } else {
              result[0] += 0.057420652;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 240))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 202))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 296))) {
              result[0] += 0.005736035;
            } else {
              result[0] += -0.018226644;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 356))) {
              result[0] += -0.014074256;
            } else {
              result[0] += -0.00035597832;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 250))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 218))) {
              result[0] += -0.00022620684;
            } else {
              result[0] += 0.021930484;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 212))) {
              result[0] += -0.00031011814;
            } else {
              result[0] += 0.018323947;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
        result[0] += 0.029248917;
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 292))) {
          result[0] += -0.014892972;
        } else {
          result[0] += 0.017006356;
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 294))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 252))) {
              result[0] += 0.0011380663;
            } else {
              result[0] += 0.0074514025;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 354))) {
              result[0] += -0.027498512;
            } else {
              result[0] += -0.00029022736;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 256))) {
            result[0] += 0.034170583;
          } else {
            result[0] += 0.022245234;
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 328))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 324))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 298))) {
              result[0] += -0.027805215;
            } else {
              result[0] += -0.017054409;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 294))) {
              result[0] += -0.016550997;
            } else {
              result[0] += -0.0043134186;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 346))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
              result[0] += -0.00029596835;
            } else {
              result[0] += -0.030671025;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
              result[0] += 0.022249522;
            } else {
              result[0] += -0.014686073;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 400))) {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 116))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 16))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
              result[0] += -0.014927131;
            } else {
              result[0] += -0.0050526834;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += 0.0031292983;
            } else {
              result[0] += -0.0021366577;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 14))) {
            result[0] += -0.019448534;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
              result[0] += 0.031444933;
            } else {
              result[0] += 0.010930794;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 36))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 152))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 20))) {
              result[0] += -0.017589707;
            } else {
              result[0] += -0.00019221655;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += -0.020101383;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 170))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
              result[0] += 0.0022330189;
            } else {
              result[0] += 0.012290857;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 14))) {
              result[0] += -0.0006709326;
            } else {
              result[0] += 0.0086814575;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 392))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 156))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 154))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0;
            } else {
              result[0] += 0.067005776;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 102))) {
              result[0] += 0.07170707;
            } else {
              result[0] += -0.014048599;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 166))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 164))) {
              result[0] += -0.030913204;
            } else {
              result[0] += -0.007337024;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 160))) {
              result[0] += 0.065959476;
            } else {
              result[0] += 0.0030737682;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 176))) {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 112))) {
            result[0] += -0.016045328;
          } else {
            result[0] += -0.008878782;
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 118))) {
              result[0] += 0.00972886;
            } else {
              result[0] += 0.03835055;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 130))) {
              result[0] += -0.009170124;
            } else {
              result[0] += 0.0057587116;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 224))) {
      result[0] += -0.015629305;
    } else {
      if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 142))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 140))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.004336093;
            } else {
              result[0] += 0.015490188;
            }
          } else {
            result[0] += -0.01513085;
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
            result[0] += 0.00940407;
          } else {
            result[0] += 0.032563876;
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 178))) {
          result[0] += -0.01413842;
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 244))) {
              result[0] += 0.0270089;
            } else {
              result[0] += -0.00225509;
            }
          } else {
            result[0] += -0.008365121;
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 110))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 168))) {
              result[0] += -0.0023444246;
            } else {
              result[0] += 0.007994309;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 16))) {
              result[0] += 0.00061645266;
            } else {
              result[0] += -0.04576966;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 62))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 20))) {
              result[0] += 0.01534583;
            } else {
              result[0] += 0.039503228;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 96))) {
              result[0] += 0.01222286;
            } else {
              result[0] += -0.0050080977;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 92))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 90))) {
              result[0] += -0.00016676041;
            } else {
              result[0] += 0.045995515;
            }
          } else {
            result[0] += -0.019292504;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 270))) {
              result[0] += -0.0106395595;
            } else {
              result[0] += 0.024322292;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 196))) {
              result[0] += 0.0038009014;
            } else {
              result[0] += -0.006002572;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 112))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 104))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 90))) {
              result[0] += 0.00059230387;
            } else {
              result[0] += 0.02190473;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
              result[0] += -0.0012604637;
            } else {
              result[0] += -0.025116524;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 46))) {
              result[0] += 0.01818419;
            } else {
              result[0] += -0.0041689784;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 54))) {
              result[0] += -0.0046932534;
            } else {
              result[0] += 0.012534521;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 100))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 212))) {
              result[0] += -0.0053321766;
            } else {
              result[0] += 3.2746604e-05;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 160))) {
              result[0] += 0.04412394;
            } else {
              result[0] += -0.019410737;
            }
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 148))) {
            result[0] += -0.015239002;
          } else {
            result[0] += -0.007977055;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 134))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 34))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 170))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 150))) {
              result[0] += 0.026527822;
            } else {
              result[0] += -0.015368457;
            }
          } else {
            result[0] += -0.05488323;
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 240))) {
            result[0] += -0.07350321;
          } else {
            result[0] += -0.01900329;
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 12))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 96))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
              result[0] += 0.0044366815;
            } else {
              result[0] += 0.05480617;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
              result[0] += -0.011877809;
            } else {
              result[0] += 0.01703535;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 106))) {
              result[0] += -0.02123229;
            } else {
              result[0] += -0.07076668;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 242))) {
              result[0] += -0.0073331865;
            } else {
              result[0] += -0.039508924;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 32))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 108))) {
            result[0] += 0.028880654;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 8))) {
              result[0] += -0.015295302;
            } else {
              result[0] += 0.0052049416;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 100))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
              result[0] += -0.0003331708;
            } else {
              result[0] += 0.025730435;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 280))) {
              result[0] += -0.0062754015;
            } else {
              result[0] += 0.013416797;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 82))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 4))) {
            result[0] += -0;
          } else {
            result[0] += -0.031346504;
          }
        } else {
          result[0] += 0.0021177256;
        }
      }
    }
  }
  if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 314))) {
    if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 240))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
              result[0] += -0.0009686186;
            } else {
              result[0] += -0.027624989;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 222))) {
              result[0] += 0.012920478;
            } else {
              result[0] += 0.03591418;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 262))) {
              result[0] += -0.027306026;
            } else {
              result[0] += -0.010613002;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += 0.028817087;
            } else {
              result[0] += -0.021536713;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 218))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 150))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 60))) {
              result[0] += -0.014519043;
            } else {
              result[0] += -0.0013858235;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
              result[0] += 0.011594709;
            } else {
              result[0] += -0.0014000179;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 312))) {
              result[0] += -0.005692505;
            } else {
              result[0] += 0.009161465;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
              result[0] += -3.463562e-05;
            } else {
              result[0] += 0.004579789;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 216))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 296))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 260))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 164))) {
              result[0] += -0.00439987;
            } else {
              result[0] += -0.020940205;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
              result[0] += -0.062401365;
            } else {
              result[0] += -0.011520316;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 150))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 326))) {
              result[0] += 0.016910944;
            } else {
              result[0] += 0.002540021;
            }
          } else {
            result[0] += -0.010653024;
          }
        }
      } else {
        result[0] += -0.056521356;
      }
    }
  } else {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 384))) {
      result[0] += 0.03409591;
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 34))) {
          result[0] += -0;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 386))) {
            result[0] += -0.0031993948;
          } else {
            result[0] += -0.02678071;
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 254))) {
          result[0] += 0.031704232;
        } else {
          result[0] += 0.005365833;
        }
      }
    }
  }
  if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 52))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 6))) {
              result[0] += 0.0014249064;
            } else {
              result[0] += -0.01947671;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 94))) {
              result[0] += 0.026210293;
            } else {
              result[0] += 0.003069733;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
            result[0] += -0.02217819;
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 40))) {
              result[0] += 0.023008201;
            } else {
              result[0] += 0.0086527765;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 14))) {
              result[0] += 0.028070768;
            } else {
              result[0] += -0;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 16))) {
              result[0] += -0.025724366;
            } else {
              result[0] += -0.008844643;
            }
          }
        } else {
          result[0] += -0.047070276;
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 134))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 44))) {
              result[0] += 0.004072014;
            } else {
              result[0] += -0.0108394185;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 170))) {
              result[0] += 0.03304889;
            } else {
              result[0] += 0.0103535205;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 114))) {
            result[0] += -0.023122396;
          } else {
            result[0] += -0.010224661;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 86))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 72))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 178))) {
              result[0] += 0.001644756;
            } else {
              result[0] += -0.002623936;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
              result[0] += -0.0045392714;
            } else {
              result[0] += -0.025003826;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 98))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 318))) {
              result[0] += 0.015446856;
            } else {
              result[0] += -0.034815937;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 34))) {
              result[0] += -0.00036249825;
            } else {
              result[0] += 0.004029076;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 86))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 48))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 76))) {
            result[0] += 0.0035753883;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 160))) {
              result[0] += -0.046707038;
            } else {
              result[0] += -0.026563475;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 136))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 48))) {
              result[0] += 0.011157422;
            } else {
              result[0] += 0.025159726;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 228))) {
              result[0] += -0.0022385176;
            } else {
              result[0] += 0.009932112;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 214))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 186))) {
              result[0] += -0.0044850186;
            } else {
              result[0] += 0.007804101;
            }
          } else {
            result[0] += 0.023426604;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 312))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 90))) {
              result[0] += -0.0033166704;
            } else {
              result[0] += -0.010364393;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 142))) {
              result[0] += -0.013235124;
            } else {
              result[0] += 0.015832936;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 104))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 98))) {
            result[0] += 0.022907645;
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 158))) {
              result[0] += 0.00681618;
            } else {
              result[0] += -0.012386585;
            }
          }
        } else {
          result[0] += 0.04736806;
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
              result[0] += -0.0017811867;
            } else {
              result[0] += 0.024726752;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 300))) {
              result[0] += -0.024868438;
            } else {
              result[0] += -0.008288312;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 86))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 300))) {
              result[0] += 0.023393013;
            } else {
              result[0] += 0.000140571;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
              result[0] += -0.0001585513;
            } else {
              result[0] += 0.0125155775;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 400))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 136))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 106))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 100))) {
              result[0] += 0.00032556072;
            } else {
              result[0] += 0.02669864;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 144))) {
              result[0] += -0.01502426;
            } else {
              result[0] += -0.007827666;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 108))) {
            result[0] += -0.042772368;
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
              result[0] += 0.0073166452;
            } else {
              result[0] += -0.003959548;
            }
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 168))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 104))) {
              result[0] += 0.00850385;
            } else {
              result[0] += -0.0007850647;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 174))) {
              result[0] += 0.022124942;
            } else {
              result[0] += 0.0037423302;
            }
          }
        } else {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 12))) {
            result[0] += -0.03591499;
          } else {
            result[0] += 0.04963535;
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 224))) {
        result[0] += -0.014330697;
      } else {
        if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 142))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 140))) {
              result[0] += 0.00258422;
            } else {
              result[0] += -0.011271856;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
              result[0] += 0.008900406;
            } else {
              result[0] += 0.025473619;
            }
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 226))) {
              result[0] += -0.013987194;
            } else {
              result[0] += -0.0071587036;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 404))) {
              result[0] += -0.007607888;
            } else {
              result[0] += 0.021506222;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 164))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
              result[0] += 0.017460436;
            } else {
              result[0] += -0.0049960585;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 48))) {
              result[0] += 0.03522423;
            } else {
              result[0] += 0.0048457957;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 92))) {
            result[0] += -0.025125725;
          } else {
            result[0] += -0;
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 4))) {
          result[0] += -0.058159776;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 292))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 80))) {
              result[0] += -0.0017305056;
            } else {
              result[0] += -0.01667433;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 306))) {
              result[0] += -0.027741238;
            } else {
              result[0] += 0.009779043;
            }
          }
        }
      }
    } else {
      result[0] += 0.009787267;
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 342))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 136))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 218))) {
              result[0] += -0.00073015277;
            } else {
              result[0] += 0.0008299645;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 146))) {
              result[0] += -0.003797351;
            } else {
              result[0] += -0.026151488;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 234))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 164))) {
              result[0] += -0.0062180604;
            } else {
              result[0] += -0.018368756;
            }
          } else {
            result[0] += 0.03389421;
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 202))) {
          result[0] += -0.001612652;
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 242))) {
            result[0] += 0.029014328;
          } else {
            result[0] += 0.018497443;
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 366))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 298))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 326))) {
            result[0] += -0.02712582;
          } else {
            result[0] += -0.014455877;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 296))) {
            result[0] += -0.017788123;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 306))) {
              result[0] += 0.003964351;
            } else {
              result[0] += -0.0068141944;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 334))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 304))) {
            result[0] += 0.046189606;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
              result[0] += 0.024067063;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 56))) {
            result[0] += 0.002387124;
          } else {
            result[0] += -0.021191007;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 316))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 48))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 140))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
              result[0] += 0.017918406;
            } else {
              result[0] += 0.0007088012;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 90))) {
              result[0] += 0.008024058;
            } else {
              result[0] += 0.03525953;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 50))) {
            result[0] += -0.017354771;
          } else {
            result[0] += 0.011550955;
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
          result[0] += 0.016495919;
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 50))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 82))) {
              result[0] += 0.0069151595;
            } else {
              result[0] += -0.009860008;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
              result[0] += -0.02526379;
            } else {
              result[0] += -0.010914913;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 84))) {
        result[0] += 0.023363909;
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 188))) {
              result[0] += -0.017829457;
            } else {
              result[0] += 0.0066434382;
            }
          } else {
            result[0] += 0.031978313;
          }
        } else {
          result[0] += -0.026702777;
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 18))) {
    if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 72))) {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 0))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 28))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 8))) {
              result[0] += 0.0016082926;
            } else {
              result[0] += -0.01875299;
            }
          } else {
            result[0] += 0.0011138814;
          }
        } else {
          result[0] += -0.05503887;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 116))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += -0.034357756;
            } else {
              result[0] += -0.00018943613;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 10))) {
              result[0] += -0.015174734;
            } else {
              result[0] += 0.006123707;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 54))) {
              result[0] += -0;
            } else {
              result[0] += -0.032295134;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 130))) {
              result[0] += 0.022839814;
            } else {
              result[0] += -0.0064965202;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 114))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 6))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 78))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 10))) {
              result[0] += -0;
            } else {
              result[0] += -0.025930112;
            }
          } else {
            result[0] += -0.040734638;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 28))) {
              result[0] += 0.0056574023;
            } else {
              result[0] += 0.045485523;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 30))) {
              result[0] += -0.014559778;
            } else {
              result[0] += -0.0022656727;
            }
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 178))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 22))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
              result[0] += 0.009212374;
            } else {
              result[0] += -0.023791295;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 122))) {
              result[0] += -0;
            } else {
              result[0] += -0.030348688;
            }
          }
        } else {
          result[0] += 0.011739956;
        }
      }
    }
  } else {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 120))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 116))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 318))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 80))) {
              result[0] += 0.00070217456;
            } else {
              result[0] += -0.01572331;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 296))) {
              result[0] += 0.0059441226;
            } else {
              result[0] += -0.015418323;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 86))) {
              result[0] += 0.0038127878;
            } else {
              result[0] += -0.0063350433;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
              result[0] += 0.023485564;
            } else {
              result[0] += -0.0005745096;
            }
          }
        }
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 28))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 396))) {
            result[0] += -0.008549287;
          } else {
            result[0] += 0.03102727;
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 76))) {
            result[0] += 0.03656302;
          } else {
            result[0] += 0.0037815608;
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
        result[0] += 0.036549788;
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 80))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 56))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 202))) {
              result[0] += -0.0316082;
            } else {
              result[0] += -0.011285119;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.044548456;
            } else {
              result[0] += -0.011524561;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.0036371031;
            } else {
              result[0] += 0.0054557663;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 250))) {
              result[0] += -0.010413558;
            } else {
              result[0] += -0.005399991;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
    if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 244))) {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 92))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 26))) {
          result[0] += -0.02938572;
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 36))) {
              result[0] += -0.0011291165;
            } else {
              result[0] += -0.023937851;
            }
          } else {
            result[0] += 0.014182518;
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 48))) {
          result[0] += 0.09786916;
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 46))) {
              result[0] += -0.039628785;
            } else {
              result[0] += -0;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 174))) {
              result[0] += 0.008892446;
            } else {
              result[0] += -0.002574872;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
        result[0] += 0.0067228116;
      } else {
        result[0] += 0.039215095;
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 170))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 106))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 50))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
              result[0] += -0.007054515;
            } else {
              result[0] += 0.035221007;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 274))) {
              result[0] += -0.030953402;
            } else {
              result[0] += -0.008404141;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 146))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 106))) {
              result[0] += -0.011643419;
            } else {
              result[0] += -0.05229321;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 288))) {
              result[0] += -0.016430408;
            } else {
              result[0] += -0.039948575;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
            result[0] += -0.015282759;
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 80))) {
              result[0] += 0.007892172;
            } else {
              result[0] += 0.078224584;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 148))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 4))) {
              result[0] += -0.0037962466;
            } else {
              result[0] += -0.015912866;
            }
          } else {
            result[0] += 0.013302639;
          }
        }
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 340))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
              result[0] += -0.00012223067;
            } else {
              result[0] += -0.012624851;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 256))) {
              result[0] += 0.027083063;
            } else {
              result[0] += 0.015543193;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 20))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 298))) {
              result[0] += -0.020818194;
            } else {
              result[0] += -0.006904413;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 320))) {
              result[0] += 0.049202304;
            } else {
              result[0] += 0.009114965;
            }
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 34))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 82))) {
              result[0] += 0.003950732;
            } else {
              result[0] += -0.011314066;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
              result[0] += 0.0068105734;
            } else {
              result[0] += 0.028154967;
            }
          }
        } else {
          result[0] += -0.021689637;
        }
      }
    }
  }
  if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 296))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 232))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 6))) {
              result[0] += -0.004839664;
            } else {
              result[0] += 0.00013240986;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 194))) {
              result[0] += 0.012408934;
            } else {
              result[0] += -0.00019027379;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 318))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 260))) {
              result[0] += -0.0021018793;
            } else {
              result[0] += 0.013699087;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 262))) {
              result[0] += -0.046322387;
            } else {
              result[0] += -0;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 260))) {
          result[0] += 0.02145579;
        } else {
          result[0] += 0.010288753;
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 358))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 286))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 112))) {
            result[0] += -0.026122052;
          } else {
            result[0] += -0.014107632;
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 12))) {
            result[0] += -0.004089996;
          } else {
            result[0] += -0.017688856;
          }
        }
      } else {
        result[0] += 0.021606075;
      }
    }
  } else {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 12))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 326))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 200))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 166))) {
              result[0] += -0.00194148;
            } else {
              result[0] += -0.012741702;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 210))) {
              result[0] += 0.036462914;
            } else {
              result[0] += -0.008751659;
            }
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 102))) {
            result[0] += -0.0035663412;
          } else {
            result[0] += -0.02093569;
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 248))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 0))) {
            result[0] += 0.014931956;
          } else {
            result[0] += 0.038147915;
          }
        } else {
          result[0] += 0.0034323465;
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 14))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 88))) {
          result[0] += -0;
        } else {
          result[0] += 0.02190816;
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 60))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 54))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += 0.0031788598;
            } else {
              result[0] += -0.017724153;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 46))) {
              result[0] += 0.00170696;
            } else {
              result[0] += -0.021163156;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 62))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 186))) {
              result[0] += -0.015714565;
            } else {
              result[0] += 0.055118978;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 346))) {
              result[0] += 0.0018268081;
            } else {
              result[0] += 0.012591888;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 258))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 144))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 98))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.005291102;
            } else {
              result[0] += -0.029125659;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 232))) {
              result[0] += 0.003732911;
            } else {
              result[0] += -0.018547367;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 240))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 152))) {
              result[0] += 0.0017839748;
            } else {
              result[0] += -0.008801985;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 118))) {
              result[0] += 0.0042691375;
            } else {
              result[0] += 0.03554597;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 208))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 224))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 284))) {
              result[0] += -0.003548905;
            } else {
              result[0] += -0.012224059;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 226))) {
              result[0] += 0.0020274771;
            } else {
              result[0] += 0.019364875;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 180))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 10))) {
              result[0] += 0.016102543;
            } else {
              result[0] += -0.013677455;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 248))) {
              result[0] += -0.05339232;
            } else {
              result[0] += -0.00876738;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 106))) {
        result[0] += -0.013133674;
      } else {
        result[0] += -0.0521831;
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 38))) {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
          result[0] += 0.008359418;
        } else {
          result[0] += 0.038568888;
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 102))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 60))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 122))) {
              result[0] += 0.0013149813;
            } else {
              result[0] += -0.0191629;
            }
          } else {
            result[0] += -0.030204315;
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 54))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 78))) {
              result[0] += 0.024884596;
            } else {
              result[0] += 0.0094683515;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 50))) {
              result[0] += 0.0068674767;
            } else {
              result[0] += -0.007323086;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 46))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 94))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 104))) {
              result[0] += -0.0076974086;
            } else {
              result[0] += -0.029438093;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 110))) {
              result[0] += 0.03408525;
            } else {
              result[0] += -0.006439855;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 62))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 76))) {
              result[0] += 0.0068958416;
            } else {
              result[0] += 0.04007944;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 84))) {
              result[0] += 0.00047964975;
            } else {
              result[0] += 0.0132646635;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 46))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 88))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 86))) {
              result[0] += 0.0022136592;
            } else {
              result[0] += 0.026103059;
            }
          } else {
            result[0] += -0.0099972095;
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 120))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
              result[0] += -0.011941642;
            } else {
              result[0] += 0.0021292144;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 20))) {
              result[0] += -0.014601464;
            } else {
              result[0] += -0.0005000638;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 250))) {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 224))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 306))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 246))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 196))) {
              result[0] += 0.000114969465;
            } else {
              result[0] += -0.0023860415;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 14))) {
              result[0] += -0.035199355;
            } else {
              result[0] += -0.017500581;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 314))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.00627641;
            } else {
              result[0] += -0.0026602177;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 292))) {
              result[0] += -0.008631534;
            } else {
              result[0] += 0.008578693;
            }
          }
        }
      } else {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 30))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 216))) {
            result[0] += 0.0028546187;
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 14))) {
              result[0] += 0.014946878;
            } else {
              result[0] += 0.041622277;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 250))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
              result[0] += 0.030365026;
            } else {
              result[0] += 0.005026304;
            }
          } else {
            result[0] += -0.030540321;
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 130))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
          result[0] += -0.034515306;
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 80))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 150))) {
              result[0] += 0.04119061;
            } else {
              result[0] += 0.018316824;
            }
          } else {
            result[0] += -0.017891763;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 72))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 54))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 200))) {
              result[0] += -0.028790927;
            } else {
              result[0] += -0.0069620805;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.04099756;
            } else {
              result[0] += -0.003357871;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 260))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 118))) {
              result[0] += -0.0026708224;
            } else {
              result[0] += -0.011060624;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
              result[0] += 0.0011356926;
            } else {
              result[0] += 0.041363157;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 190))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 188))) {
        result[0] += 0.024280345;
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 320))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 0))) {
            result[0] += 0.010236905;
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 182))) {
              result[0] += -0.02040131;
            } else {
              result[0] += -0.008407633;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 326))) {
            result[0] += 0.012552517;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
              result[0] += -0.021955565;
            } else {
              result[0] += -0.0035318558;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 120))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 302))) {
              result[0] += 0.009495072;
            } else {
              result[0] += 0.00013142404;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 248))) {
              result[0] += -0.006438227;
            } else {
              result[0] += -0.030158589;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 306))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 276))) {
              result[0] += -0;
            } else {
              result[0] += 0.018983908;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 60))) {
              result[0] += -0.044639092;
            } else {
              result[0] += 0.00017055031;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 60))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 354))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 346))) {
              result[0] += -0.008808951;
            } else {
              result[0] += -0.032180067;
            }
          } else {
            result[0] += 0.008250392;
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 62))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 56))) {
              result[0] += 0.056063425;
            } else {
              result[0] += -0.0065511647;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
              result[0] += -0.0092850765;
            } else {
              result[0] += 0.0021918796;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 30))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
              result[0] += 0.007402791;
            } else {
              result[0] += -0.011684748;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += -0.006889455;
            } else {
              result[0] += 0.01800547;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 36))) {
            result[0] += -0.024754832;
          } else {
            result[0] += -0.0028222399;
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 272))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 30))) {
              result[0] += -0.020031106;
            } else {
              result[0] += 0.0067768777;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 52))) {
              result[0] += 0.041279774;
            } else {
              result[0] += 0.012748748;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 144))) {
            result[0] += -0;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 84))) {
              result[0] += 0.015250462;
            } else {
              result[0] += 0.026752282;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 250))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 216))) {
            result[0] += -0.02535589;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 156))) {
              result[0] += 0.006471269;
            } else {
              result[0] += 0.029563783;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 170))) {
            result[0] += -0.025211362;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 186))) {
              result[0] += -0.023914715;
            } else {
              result[0] += -0.009117383;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 154))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
            result[0] += 0.0012569004;
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 118))) {
              result[0] += -0.008115943;
            } else {
              result[0] += -0.026224598;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 340))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 196))) {
              result[0] += 0.042526796;
            } else {
              result[0] += 0.022222739;
            }
          } else {
            result[0] += 0.012392859;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 12))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 40))) {
          result[0] += -0.028314574;
        } else {
          result[0] += -0.002670089;
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
          result[0] += 0.0057236534;
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 120))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 50))) {
              result[0] += -0.017261336;
            } else {
              result[0] += -0.007837756;
            }
          } else {
            result[0] += -0.0055499817;
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 216))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 78))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 56))) {
              result[0] += -0.0009610457;
            } else {
              result[0] += 0.0076379874;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 16))) {
              result[0] += 0.05212106;
            } else {
              result[0] += 0.02506253;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
              result[0] += -0.0006034739;
            } else {
              result[0] += 0.021560185;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
              result[0] += -0.034164276;
            } else {
              result[0] += -0.004662303;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 78))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
              result[0] += -0.001952683;
            } else {
              result[0] += 0.037713937;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 152))) {
              result[0] += -0.008823862;
            } else {
              result[0] += -0.0014422481;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 76))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 242))) {
              result[0] += 0.014311755;
            } else {
              result[0] += -0.018534161;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
              result[0] += 0.0027180712;
            } else {
              result[0] += -0.0037262347;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 152))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 306))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 296))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 46))) {
              result[0] += -0.00358946;
            } else {
              result[0] += 0.00086048665;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 298))) {
              result[0] += -0.020779364;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 184))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 34))) {
              result[0] += -0.039312568;
            } else {
              result[0] += -0.005182448;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 82))) {
              result[0] += 0.0015322726;
            } else {
              result[0] += -0.0016843865;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 218))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 312))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
              result[0] += -0.006095136;
            } else {
              result[0] += 0.028541317;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 188))) {
              result[0] += 0.015441231;
            } else {
              result[0] += -0.01365337;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
              result[0] += 0.028764328;
            } else {
              result[0] += -0;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
              result[0] += 0.01359984;
            } else {
              result[0] += -0.0053497027;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 118))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 216))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 254))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 38))) {
              result[0] += 0.0073710456;
            } else {
              result[0] += 0.00038945695;
            }
          } else {
            result[0] += -0.00408677;
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 60))) {
            result[0] += -0.016966945;
          } else {
            result[0] += -0;
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 130))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 106))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 70))) {
              result[0] += 0.0059221475;
            } else {
              result[0] += 0.018299853;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
              result[0] += 0.005721331;
            } else {
              result[0] += -0.010450677;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 214))) {
            result[0] += 0.03640103;
          } else {
            result[0] += 0.01191198;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 96))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 88))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 70))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 76))) {
              result[0] += -0.036111545;
            } else {
              result[0] += -0;
            }
          } else {
            result[0] += 0.013476851;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 74))) {
            result[0] += -0.02820078;
          } else {
            result[0] += -0.08100428;
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 76))) {
          result[0] += 0.001866552;
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 48))) {
            result[0] += -0.021744216;
          } else {
            result[0] += -0;
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 48))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 42))) {
              result[0] += -0.004817111;
            } else {
              result[0] += -0.019636374;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 70))) {
              result[0] += 0.017743232;
            } else {
              result[0] += -0.010167501;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 66))) {
            result[0] += -0.05105884;
          } else {
            result[0] += -0.011944066;
          }
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 102))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 68))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 24))) {
              result[0] += -0;
            } else {
              result[0] += 0.02911661;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 66))) {
              result[0] += -0.010146369;
            } else {
              result[0] += 0.009323521;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 2))) {
            result[0] += -0.03494624;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
              result[0] += -0.003336961;
            } else {
              result[0] += -0.02113694;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 92))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 90))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 166))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 114))) {
              result[0] += -0.0013070774;
            } else {
              result[0] += 0.0005617281;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 356))) {
              result[0] += -0.00045852616;
            } else {
              result[0] += 0.021694412;
            }
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 190))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 54))) {
              result[0] += -0.009019866;
            } else {
              result[0] += 0.0010552766;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 382))) {
              result[0] += 0.042176425;
            } else {
              result[0] += -0.019407976;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 54))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 94))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 22))) {
              result[0] += -0.008323395;
            } else {
              result[0] += -0.0172386;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 96))) {
              result[0] += 0.0036288549;
            } else {
              result[0] += -0.013044089;
            }
          }
        } else {
          result[0] += 0.00021139935;
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 392))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 158))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 74))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 78))) {
              result[0] += -0.00079369405;
            } else {
              result[0] += 0.04720639;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 390))) {
              result[0] += 0.05325351;
            } else {
              result[0] += -0.043466292;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 66))) {
            result[0] += -0.032283835;
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 198))) {
              result[0] += -0.0008110626;
            } else {
              result[0] += 0.050249096;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 68))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 158))) {
            result[0] += -0.010387595;
          } else {
            result[0] += -0.0034609255;
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 110))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 394))) {
              result[0] += 0.0030221925;
            } else {
              result[0] += 0.028211419;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 114))) {
              result[0] += -0.0093119275;
            } else {
              result[0] += 0.0069152773;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 164))) {
      result[0] += -0.009060017;
    } else {
      if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 128))) {
        if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 126))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 46))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 10))) {
              result[0] += -0.006120693;
            } else {
              result[0] += 0.009823619;
            }
          } else {
            result[0] += -0.009353681;
          }
        } else {
          result[0] += -0.0127736945;
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 400))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 16))) {
            result[0] += 0.013670566;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 86))) {
              result[0] += -0.004946747;
            } else {
              result[0] += 0.0048705908;
            }
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 136))) {
            result[0] += -0.010391137;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 92))) {
              result[0] += 0.004601255;
            } else {
              result[0] += -0.0049207225;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 22))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 112))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
              result[0] += -0.018509652;
            } else {
              result[0] += -0.0028016684;
            }
          } else {
            result[0] += 0.012803587;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 14))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.013896561;
            } else {
              result[0] += -0.009757071;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 2))) {
              result[0] += 0.035226237;
            } else {
              result[0] += 0.003883094;
            }
          }
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 108))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 72))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
              result[0] += 0.009042918;
            } else {
              result[0] += 0.03158817;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 138))) {
              result[0] += 0.007958132;
            } else {
              result[0] += 0.03468901;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 146))) {
            result[0] += -0.0020058847;
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 174))) {
              result[0] += 0.026864871;
            } else {
              result[0] += 0.014916946;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 250))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 216))) {
            result[0] += -0.022077216;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 234))) {
              result[0] += 0.007497831;
            } else {
              result[0] += 0.028885541;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 250))) {
            result[0] += -0.02360869;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 174))) {
              result[0] += -0.018704783;
            } else {
              result[0] += -0.0076525756;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 294))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
              result[0] += 9.27179e-05;
            } else {
              result[0] += 0.031941853;
            }
          } else {
            result[0] += 0.009628982;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 350))) {
            result[0] += -0.021527562;
          } else {
            result[0] += -0.0072527933;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 74))) {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 104))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 70))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 40))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 60))) {
              result[0] += 0.02288001;
            } else {
              result[0] += -0.016261838;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 146))) {
              result[0] += -0.03769898;
            } else {
              result[0] += -0.018141152;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 46))) {
              result[0] += -0.0023420886;
            } else {
              result[0] += 0.021080358;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 140))) {
              result[0] += -0.012038226;
            } else {
              result[0] += 0.008399843;
            }
          }
        }
      } else {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 72))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += 0.002315519;
            } else {
              result[0] += 0.062828414;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
              result[0] += 0.0130506335;
            } else {
              result[0] += -0.007596428;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 356))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 40))) {
              result[0] += 0.007198368;
            } else {
              result[0] += -0.0058715134;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
              result[0] += 0.02399823;
            } else {
              result[0] += -0.008211115;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 84))) {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 26))) {
              result[0] += 0.02461793;
            } else {
              result[0] += 0.008668258;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 28))) {
              result[0] += -0.030365562;
            } else {
              result[0] += 0.010669919;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 94))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 16))) {
              result[0] += -0.018534234;
            } else {
              result[0] += -0.0031345394;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += -0.010914135;
            } else {
              result[0] += -0.00012393964;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 214))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 88))) {
              result[0] += 0.022872802;
            } else {
              result[0] += 0.0050913426;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 132))) {
              result[0] += 0.0009941938;
            } else {
              result[0] += -0.0025577117;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += 0.012239212;
            } else {
              result[0] += 0.032537233;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 74))) {
              result[0] += -0.012231956;
            } else {
              result[0] += 0.00096511666;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 244))) {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 50))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 100))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 190))) {
              result[0] += -0.015980741;
            } else {
              result[0] += 0.013686332;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 106))) {
              result[0] += 0.030151868;
            } else {
              result[0] += 0.008428541;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 20))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 192))) {
              result[0] += 0.001345482;
            } else {
              result[0] += 0.02078061;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 126))) {
              result[0] += -0.029191677;
            } else {
              result[0] += 0.00064531795;
            }
          }
        }
      } else {
        result[0] += 0.035184488;
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 106))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 30))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 212))) {
              result[0] += 0.009157064;
            } else {
              result[0] += -0.03418844;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
              result[0] += -0.01716392;
            } else {
              result[0] += -0.045232523;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
              result[0] += -0.01179669;
            } else {
              result[0] += 0.06294518;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 140))) {
              result[0] += -0.011332896;
            } else {
              result[0] += 0.01226867;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 134))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
              result[0] += 0.0011395406;
            } else {
              result[0] += -0.0025050612;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 138))) {
              result[0] += -0.037857305;
            } else {
              result[0] += -0.00218284;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 226))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 102))) {
              result[0] += -0.0008166955;
            } else {
              result[0] += 0.004205488;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.0056326287;
            } else {
              result[0] += 0.0007912867;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 6))) {
      result[0] += 0.023435187;
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 40))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 42))) {
            result[0] += 0.03141637;
          } else {
            result[0] += -0.0015305742;
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 268))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 8))) {
              result[0] += -0.017991172;
            } else {
              result[0] += 0.002195324;
            }
          } else {
            result[0] += 0.013269196;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 44))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 146))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 120))) {
              result[0] += -0.01257918;
            } else {
              result[0] += 0.0010349855;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 154))) {
              result[0] += -0.051812388;
            } else {
              result[0] += -0.010208193;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 30))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 278))) {
              result[0] += 0.0073652193;
            } else {
              result[0] += -0.013559197;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 56))) {
              result[0] += -0.02485094;
            } else {
              result[0] += -0.0037162665;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 12))) {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
      if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 42))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 2))) {
              result[0] += -0.034541044;
            } else {
              result[0] += 0.0027651105;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 6))) {
              result[0] += -0.019249316;
            } else {
              result[0] += -0.0034680355;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 8))) {
            result[0] += -0.02443984;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += 0.013286717;
            } else {
              result[0] += -0.0025256465;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 46))) {
          result[0] += 0.0061984803;
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 48))) {
            result[0] += 0.079341814;
          } else {
            result[0] += 0.025329694;
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 6))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 126))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 14))) {
            result[0] += 0.021308701;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 118))) {
              result[0] += -0.006669204;
            } else {
              result[0] += -0.019334236;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 140))) {
            result[0] += 0.04223214;
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 74))) {
              result[0] += -0.01822163;
            } else {
              result[0] += 0.011897653;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 58))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 80))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 12))) {
              result[0] += -0.03937181;
            } else {
              result[0] += -0.021595992;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 58))) {
              result[0] += 0.008440125;
            } else {
              result[0] += -0.018780667;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 32))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
              result[0] += -0.0025710927;
            } else {
              result[0] += -0.015579129;
            }
          } else {
            result[0] += 0.013569686;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 118))) {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 10))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 102))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 90))) {
              result[0] += -0.008451057;
            } else {
              result[0] += 0.010793014;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 244))) {
              result[0] += -0.007276828;
            } else {
              result[0] += -0.00065764156;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 130))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
              result[0] += 0.00065685576;
            } else {
              result[0] += 0.025356445;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 142))) {
              result[0] += -0.012041635;
            } else {
              result[0] += -0.05446145;
            }
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 84))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 120))) {
              result[0] += 0.044229705;
            } else {
              result[0] += 0.0006534785;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 30))) {
              result[0] += 0.005015543;
            } else {
              result[0] += -0.01020183;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += 0.06759548;
            } else {
              result[0] += 0.007752572;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
              result[0] += 0.011832262;
            } else {
              result[0] += 0.0015743548;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 112))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 208))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 196))) {
              result[0] += -0.0024539565;
            } else {
              result[0] += 0.001751725;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
              result[0] += -0;
            } else {
              result[0] += 0.021780172;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 66))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 272))) {
              result[0] += 0.008948053;
            } else {
              result[0] += -0.0063331993;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
              result[0] += -0.0073422543;
            } else {
              result[0] += -0.026588783;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 210))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 104))) {
              result[0] += 0.0029568898;
            } else {
              result[0] += -0.042533282;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
              result[0] += -0.0074031386;
            } else {
              result[0] += 0.014356777;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
              result[0] += 0.00092436554;
            } else {
              result[0] += 0.024809679;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 326))) {
              result[0] += -0.00329214;
            } else {
              result[0] += 0.00526762;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 142))) {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 10))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 200))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 72))) {
              result[0] += 0.0010467954;
            } else {
              result[0] += 0.030300766;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 62))) {
              result[0] += 0.02035891;
            } else {
              result[0] += -0.010212629;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 250))) {
              result[0] += -0.001989516;
            } else {
              result[0] += -0.013169396;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 346))) {
              result[0] += 0.020909293;
            } else {
              result[0] += -0.0079391515;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 306))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 262))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 116))) {
              result[0] += -0.0052688015;
            } else {
              result[0] += 0.0019646974;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 286))) {
              result[0] += -0.031940565;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 80))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 322))) {
              result[0] += 0.020321852;
            } else {
              result[0] += 0.034146465;
            }
          } else {
            result[0] += -0.03401268;
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 84))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 120))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 118))) {
              result[0] += 0.0016832197;
            } else {
              result[0] += 0.027038325;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 166))) {
              result[0] += -0.0071495273;
            } else {
              result[0] += -0.0014786319;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 258))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
              result[0] += 0.01780109;
            } else {
              result[0] += -0.01263422;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 228))) {
              result[0] += 0.039167188;
            } else {
              result[0] += 0.018858656;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 42))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 26))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 356))) {
              result[0] += -0.0019396268;
            } else {
              result[0] += 0.005724409;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 6))) {
              result[0] += 0.027276058;
            } else {
              result[0] += 0.008047543;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 130))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
              result[0] += -0.012420956;
            } else {
              result[0] += -0.0021616125;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
              result[0] += -0.0018222294;
            } else {
              result[0] += 0.0021800934;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 150))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 92))) {
          result[0] += -0.006505067;
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 220))) {
            result[0] += 0.040462743;
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 46))) {
              result[0] += 0.015701324;
            } else {
              result[0] += -0.0201608;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 116))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 52))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 94))) {
              result[0] += 0.0005646117;
            } else {
              result[0] += 0.010826596;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += -0.013457968;
            } else {
              result[0] += -1.2906838e-05;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += -0.0008355248;
            } else {
              result[0] += -0.0226526;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 92))) {
              result[0] += 0.020389868;
            } else {
              result[0] += 0.0054127253;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 14))) {
        if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 76))) {
          result[0] += -0.032427344;
        } else {
          result[0] += -0;
        }
      } else {
        result[0] += 0.04492506;
      }
    }
  }
  if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 18))) {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
      if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 40))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 8))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
              result[0] += -0.008939014;
            } else {
              result[0] += -0.031956512;
            }
          } else {
            result[0] += 0.017092446;
          }
        } else {
          result[0] += 0.022510538;
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 4))) {
          result[0] += 0.042899013;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
            result[0] += -0.0038211767;
          } else {
            result[0] += 0.029483128;
          }
        }
      }
    } else {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 72))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 160))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 82))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
              result[0] += -0.007246212;
            } else {
              result[0] += 0.0062661204;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 88))) {
              result[0] += -0.02025948;
            } else {
              result[0] += -0.005702542;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 116))) {
            result[0] += -0.030373631;
          } else {
            result[0] += -0.004761403;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 90))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 272))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 268))) {
              result[0] += 0.010341472;
            } else {
              result[0] += 0.0385042;
            }
          } else {
            result[0] += -0.0025437989;
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 204))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
              result[0] += -0.0027436535;
            } else {
              result[0] += -0.020981843;
            }
          } else {
            result[0] += 0.011424384;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 130))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 32))) {
            result[0] += -0.026424287;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 22))) {
              result[0] += 0.015828108;
            } else {
              result[0] += -0.004946768;
            }
          }
        } else {
          result[0] += 0.036206927;
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 0))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 46))) {
            result[0] += -0.0018861439;
          } else {
            result[0] += 0.019979624;
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 16))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 64))) {
              result[0] += -0.012420944;
            } else {
              result[0] += -0.04097703;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 46))) {
              result[0] += -0.014519716;
            } else {
              result[0] += 0.0064661787;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 116))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
              result[0] += -0.0007507774;
            } else {
              result[0] += 0.011509613;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 74))) {
              result[0] += 0.0015937341;
            } else {
              result[0] += 0.026918387;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 268))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 76))) {
              result[0] += -0.001206148;
            } else {
              result[0] += -0.01964714;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 98))) {
              result[0] += 0.008147176;
            } else {
              result[0] += -0.0026112716;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 92))) {
              result[0] += -0.00044672118;
            } else {
              result[0] += -0.009276496;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 156))) {
              result[0] += 0.039460566;
            } else {
              result[0] += -0.0033604186;
            }
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 148))) {
            result[0] += -0.0112617565;
          } else {
            result[0] += -0.0053185816;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 240))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 32))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 16))) {
              result[0] += -0.027565295;
            } else {
              result[0] += -0.004120046;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
              result[0] += -0.0030114742;
            } else {
              result[0] += 0.010990894;
            }
          }
        } else {
          result[0] += -0.018377742;
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 272))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
              result[0] += 0.0032660365;
            } else {
              result[0] += 0.014772459;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 240))) {
              result[0] += 0.01916596;
            } else {
              result[0] += 0.043604817;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 188))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
              result[0] += 0.020365;
            } else {
              result[0] += -0.004660358;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 182))) {
              result[0] += -0.026131554;
            } else {
              result[0] += -0.012585315;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 208))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 144))) {
          result[0] += -0.009168918;
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 186))) {
              result[0] += -0.017280573;
            } else {
              result[0] += 0.013135247;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 294))) {
              result[0] += 0.027347783;
            } else {
              result[0] += 0.0058005466;
            }
          }
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 228))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 262))) {
            result[0] += -0.01863461;
          } else {
            result[0] += -0.009977842;
          }
        } else {
          result[0] += 0.0074623013;
        }
      }
    }
  } else {
    if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 18))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 82))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 10))) {
            result[0] += -0.027271813;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += 0.03428016;
            } else {
              result[0] += 0.005898777;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 124))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 120))) {
              result[0] += -0.0031157867;
            } else {
              result[0] += 0.012381009;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 40))) {
              result[0] += -0.032176975;
            } else {
              result[0] += -0.0047460934;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 128))) {
              result[0] += -0.011950021;
            } else {
              result[0] += -0.029780094;
            }
          } else {
            result[0] += -0.038288817;
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 14))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 58))) {
              result[0] += -0.013251813;
            } else {
              result[0] += 0.009832563;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 40))) {
              result[0] += -0.0028266981;
            } else {
              result[0] += 0.010589391;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 26))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 0))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 144))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 8))) {
              result[0] += -0;
            } else {
              result[0] += 0.02299802;
            }
          } else {
            result[0] += -0.0005336401;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 16))) {
            result[0] += 0.024464216;
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 24))) {
              result[0] += -0.02506329;
            } else {
              result[0] += -0.00617628;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 146))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 228))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 168))) {
              result[0] += 0.000546928;
            } else {
              result[0] += -0.0015107197;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 244))) {
              result[0] += 0.0052600866;
            } else {
              result[0] += 0.0005289404;
            }
          }
        } else {
          result[0] += -0.0067100944;
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 120))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 92))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 90))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 56))) {
              result[0] += 0.000776489;
            } else {
              result[0] += -0.00053459004;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.0025825037;
            } else {
              result[0] += 0.033680957;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 54))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 94))) {
              result[0] += -0.010514084;
            } else {
              result[0] += 0.00012713655;
            }
          } else {
            result[0] += -8.464566e-05;
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 46))) {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 192))) {
              result[0] += -0.008038435;
            } else {
              result[0] += 0.03207838;
            }
          } else {
            result[0] += -0.0059311604;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 196))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 104))) {
              result[0] += -0.0027816568;
            } else {
              result[0] += 0.008232006;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += 0.03486943;
            } else {
              result[0] += -0.009670003;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 78))) {
        result[0] += -0.021940826;
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 160))) {
          result[0] += -0.007266288;
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 128))) {
              result[0] += -0.004397231;
            } else {
              result[0] += 0.0011502573;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
              result[0] += -0.005171144;
            } else {
              result[0] += 0.009149543;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 132))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 68))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
          result[0] += 0.033398908;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 34))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 146))) {
              result[0] += -0.011877987;
            } else {
              result[0] += 0.009033947;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 32))) {
              result[0] += 0.017914457;
            } else {
              result[0] += 0.00086382544;
            }
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 114))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 106))) {
            result[0] += -0;
          } else {
            result[0] += -0.025332725;
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 136))) {
            result[0] += 0.010888358;
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 132))) {
              result[0] += -0.002672884;
            } else {
              result[0] += -0.0126517145;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
          result[0] += -0.01176235;
        } else {
          result[0] += -0.04480323;
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 132))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 80))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 146))) {
              result[0] += 0.022533778;
            } else {
              result[0] += 0.0015053398;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
              result[0] += -0.029711738;
            } else {
              result[0] += -0.00018641823;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 216))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 194))) {
              result[0] += -0.0029512055;
            } else {
              result[0] += -0.014446958;
            }
          } else {
            result[0] += -0.04525297;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 160))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 116))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 126))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 108))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 104))) {
              result[0] += 0.0049565816;
            } else {
              result[0] += 0.063526;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 114))) {
              result[0] += -0.019255767;
            } else {
              result[0] += 0.0012636579;
            }
          }
        } else {
          result[0] += 0.008695122;
        }
      } else {
        result[0] += 0.019830158;
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 190))) {
        result[0] += -0.007866302;
      } else {
        result[0] += 0.012916532;
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 106))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 30))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
            result[0] += 0.06366762;
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 98))) {
              result[0] += -0.004975738;
            } else {
              result[0] += -0.026275873;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 238))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 148))) {
              result[0] += -0.009829768;
            } else {
              result[0] += -0.028741485;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 290))) {
              result[0] += -0.0046568112;
            } else {
              result[0] += -0.028392253;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
            result[0] += -0.01022596;
          } else {
            result[0] += 0.057439942;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 142))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 10))) {
              result[0] += -0.005370389;
            } else {
              result[0] += -0.019819807;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 70))) {
              result[0] += 0.011075004;
            } else {
              result[0] += -0.0060161785;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 158))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 20))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
              result[0] += 0.062947415;
            } else {
              result[0] += 0.0032754538;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 146))) {
              result[0] += -0.0023154307;
            } else {
              result[0] += 0.013836692;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
              result[0] += -0.0051080496;
            } else {
              result[0] += -0.026022715;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 130))) {
              result[0] += -0.021521274;
            } else {
              result[0] += 0.0008544709;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 16))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 44))) {
            result[0] += 0.011564689;
          } else {
            result[0] += 0.039744;
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 134))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 202))) {
              result[0] += 0.00052881095;
            } else {
              result[0] += -0.0008797978;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 56))) {
              result[0] += 0.002648249;
            } else {
              result[0] += 0.02056101;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 22))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 160))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 142))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 114))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 92))) {
              result[0] += -0.0069403;
            } else {
              result[0] += 0.0036496245;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 6))) {
              result[0] += -0.041129064;
            } else {
              result[0] += -0.007193537;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 138))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 14))) {
              result[0] += -0.016804745;
            } else {
              result[0] += -0.035601314;
            }
          } else {
            result[0] += 0.0045456374;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 30))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 154))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 0))) {
              result[0] += 0.051934537;
            } else {
              result[0] += -0.0017292331;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 18))) {
              result[0] += -0.018062782;
            } else {
              result[0] += -0.0065657706;
            }
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 36))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 12))) {
              result[0] += -0.0018053079;
            } else {
              result[0] += 0.008867439;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
              result[0] += 0.024381882;
            } else {
              result[0] += 0.0078328205;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 194))) {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 126))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 118))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 60))) {
              result[0] += -0.0050182706;
            } else {
              result[0] += 0.0074505755;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 48))) {
              result[0] += -0.027363313;
            } else {
              result[0] += -0.008215737;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 238))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 132))) {
              result[0] += 0.0108621;
            } else {
              result[0] += 0.001027461;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
              result[0] += 0.0018521305;
            } else {
              result[0] += -0.019366777;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 196))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 66))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 42))) {
              result[0] += -0.021509835;
            } else {
              result[0] += -0.0051732687;
            }
          } else {
            result[0] += -0.036974095;
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 10))) {
            result[0] += 0.013641315;
          } else {
            result[0] += -0.0012665674;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 168))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 110))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 120))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 382))) {
              result[0] += -0.0007951208;
            } else {
              result[0] += -0.012272722;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 110))) {
              result[0] += 0.013259816;
            } else {
              result[0] += 0.00076070236;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 124))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 86))) {
              result[0] += -0.023466855;
            } else {
              result[0] += -0.0015092399;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
              result[0] += 0.0067120204;
            } else {
              result[0] += -0.0027905267;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 46))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 180))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 132))) {
              result[0] += 0.0012523715;
            } else {
              result[0] += 0.019288411;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 72))) {
              result[0] += -0.0024032074;
            } else {
              result[0] += 0.020834908;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 100))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 302))) {
              result[0] += -0.0045833127;
            } else {
              result[0] += 0.002748456;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
              result[0] += -0.00035413975;
            } else {
              result[0] += 0.0021194927;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 352))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 196))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 162))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 36))) {
              result[0] += 0.012191023;
            } else {
              result[0] += -0;
            }
          } else {
            result[0] += 0.026147142;
          }
        } else {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 82))) {
            result[0] += -0.0044075255;
          } else {
            result[0] += -0.018723859;
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 368))) {
          result[0] += 0.02115663;
        } else {
          result[0] += -0.008576654;
        }
      }
    }
  }
  if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 116))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 312))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 78))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 146))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 108))) {
              result[0] += -0.0011369804;
            } else {
              result[0] += 0.007111168;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
              result[0] += -0.0066426345;
            } else {
              result[0] += 0.004026822;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 166))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 126))) {
              result[0] += -0.016191296;
            } else {
              result[0] += -0.008390819;
            }
          } else {
            result[0] += 0.039027248;
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 338))) {
              result[0] += 0.011132075;
            } else {
              result[0] += 0.00045876933;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 108))) {
              result[0] += 0.03038956;
            } else {
              result[0] += 0.005012113;
            }
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 128))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 126))) {
              result[0] += -0.002966301;
            } else {
              result[0] += -0.012031978;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 400))) {
              result[0] += 0.0043450063;
            } else {
              result[0] += -0.0033799019;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 74))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 72))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 172))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 0))) {
              result[0] += 0.00018884889;
            } else {
              result[0] += 0.004433022;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 140))) {
              result[0] += -0.0016364238;
            } else {
              result[0] += 0.011106547;
            }
          }
        } else {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 34))) {
            result[0] += 0.010573897;
          } else {
            result[0] += -0.046158385;
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
          result[0] += 0.025685733;
        } else {
          result[0] += -0.012269548;
        }
      }
    }
  } else {
    if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 298))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 258))) {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 328))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 132))) {
              result[0] += 0.012633822;
            } else {
              result[0] += -0.004591215;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 68))) {
              result[0] += 0.027154177;
            } else {
              result[0] += -0.0073612966;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 90))) {
              result[0] += -0.009671869;
            } else {
              result[0] += 0.034116954;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 82))) {
              result[0] += 0.003360726;
            } else {
              result[0] += -0.0010045286;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 242))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 80))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 294))) {
              result[0] += -0.0067084306;
            } else {
              result[0] += 0.017508617;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 52))) {
              result[0] += -0.025245184;
            } else {
              result[0] += -0.009711;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 266))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 96))) {
              result[0] += 0.0024178706;
            } else {
              result[0] += 0.02038781;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
              result[0] += -0.005333036;
            } else {
              result[0] += -0.021810295;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 314))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 148))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 44))) {
            result[0] += 0.002272234;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 302))) {
              result[0] += 0.0017241904;
            } else {
              result[0] += 0.014623227;
            }
          }
        } else {
          result[0] += -0.0057367506;
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 140))) {
            result[0] += -0.029135648;
          } else {
            result[0] += -0.0059048724;
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 80))) {
            result[0] += 0.0016456817;
          } else {
            result[0] += 0.036199924;
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 314))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 312))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 306))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
              result[0] += 5.137796e-05;
            } else {
              result[0] += 0.016802512;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 344))) {
              result[0] += -0.007183667;
            } else {
              result[0] += 0.0086045405;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 290))) {
              result[0] += -0.0025281047;
            } else {
              result[0] += -0.022966359;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 250))) {
              result[0] += 0.023195243;
            } else {
              result[0] += 0.003728645;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 186))) {
          result[0] += 0.008694565;
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
            result[0] += -0.012267268;
          } else {
            result[0] += 0.012734731;
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
        result[0] += 0.023913363;
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 228))) {
            result[0] += -0.0029125137;
          } else {
            result[0] += -0.030048529;
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 226))) {
            result[0] += -0;
          } else {
            result[0] += 0.04022684;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 296))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 8))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 2))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
            result[0] += 0.010619841;
          } else {
            result[0] += -0.015812444;
          }
        } else {
          result[0] += 0.036461692;
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 10))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 92))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 10))) {
              result[0] += 0.006464646;
            } else {
              result[0] += -0.03794737;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 6))) {
              result[0] += -0.014366518;
            } else {
              result[0] += 0.00065602706;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += -0.036709867;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 26))) {
              result[0] += 0.009573085;
            } else {
              result[0] += -0.0042258967;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
        result[0] += 0.025289237;
      } else {
        result[0] += 0.0031290874;
      }
    }
  }
  if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 32))) {
    if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 314))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 176))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 16))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 38))) {
              result[0] += 0.0070433277;
            } else {
              result[0] += -0.0021675983;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.00069358293;
            } else {
              result[0] += 0.014411396;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 224))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 182))) {
              result[0] += -0.021365121;
            } else {
              result[0] += -0.009108902;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 258))) {
              result[0] += -0.002070323;
            } else {
              result[0] += -0.0070032175;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 58))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 116))) {
              result[0] += 0.015421606;
            } else {
              result[0] += 0.0065408074;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 68))) {
              result[0] += -0.00017713972;
            } else {
              result[0] += 0.0036349527;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 66))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 112))) {
              result[0] += -0.015431674;
            } else {
              result[0] += 0.0069513856;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 86))) {
              result[0] += -0.0017071629;
            } else {
              result[0] += 0.0010942215;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 60))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 328))) {
          result[0] += -0.018242605;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 114))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 334))) {
              result[0] += 0.013435415;
            } else {
              result[0] += 0.000363245;
            }
          } else {
            result[0] += -0.011918341;
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 62))) {
          result[0] += 0.057550073;
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 64))) {
            result[0] += -0.010455087;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 292))) {
              result[0] += 0.025410349;
            } else {
              result[0] += 0.0067503005;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 48))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 250))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 4))) {
              result[0] += 0.0038494274;
            } else {
              result[0] += -0.012583888;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
              result[0] += 0.0052445154;
            } else {
              result[0] += 0.022994613;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 110))) {
              result[0] += 0.010131297;
            } else {
              result[0] += 0.030005908;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 36))) {
              result[0] += -0.015509938;
            } else {
              result[0] += 0.033259317;
            }
          }
        }
      } else {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 256))) {
          result[0] += -0.018772317;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 324))) {
              result[0] += -0.0010802757;
            } else {
              result[0] += -0.019551003;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
              result[0] += 0.015505517;
            } else {
              result[0] += -0.014615647;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 228))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 266))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
              result[0] += -0.0011533442;
            } else {
              result[0] += 0.0037619993;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 42))) {
              result[0] += -0.0050083552;
            } else {
              result[0] += -0.020327281;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.01845751;
            } else {
              result[0] += -0.0027648432;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 270))) {
              result[0] += 0.014990672;
            } else {
              result[0] += -0.0010248877;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 354))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
              result[0] += -0.00588067;
            } else {
              result[0] += -0.0011169165;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
              result[0] += 0.0015550206;
            } else {
              result[0] += -0.0028203435;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 10))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 248))) {
              result[0] += 0.00057445857;
            } else {
              result[0] += -0.020817637;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 176))) {
              result[0] += 0.004375774;
            } else {
              result[0] += -0.00017658187;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 116))) {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 92))) {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 40))) {
          result[0] += -0.029479165;
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 36))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
              result[0] += -0.0015629175;
            } else {
              result[0] += 0.014696643;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 40))) {
              result[0] += -0.012695557;
            } else {
              result[0] += 0.002633101;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 38))) {
          result[0] += 0.088580444;
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 18))) {
              result[0] += -0.0013535247;
            } else {
              result[0] += -0.041627154;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 4))) {
              result[0] += 0.010081653;
            } else {
              result[0] += 0.0011864647;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 158))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
          result[0] += 0.024709804;
        } else {
          result[0] += 0.006805406;
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
          result[0] += -0.020788606;
        } else {
          result[0] += 0.010531965;
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 106))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 22))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 6))) {
              result[0] += 0.029835364;
            } else {
              result[0] += 0.072482795;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 98))) {
              result[0] += -0.0033422098;
            } else {
              result[0] += -0.027484313;
            }
          }
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 198))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 130))) {
              result[0] += -0.0117129935;
            } else {
              result[0] += -0.027794067;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 290))) {
              result[0] += -0.0058866264;
            } else {
              result[0] += -0.022589795;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 96))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
            result[0] += -0.007361448;
          } else {
            result[0] += 0.05226938;
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 152))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
              result[0] += -0.0011431018;
            } else {
              result[0] += -0.013976167;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 34))) {
              result[0] += -0;
            } else {
              result[0] += 0.0123240035;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 16))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 130))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 42))) {
              result[0] += 0.00010176472;
            } else {
              result[0] += 0.0130502;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
              result[0] += -0.00884224;
            } else {
              result[0] += -0.0399912;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 166))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 102))) {
              result[0] += -0.0070857047;
            } else {
              result[0] += 0.0050304467;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
              result[0] += -0.018932706;
            } else {
              result[0] += -0.0011952891;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 16))) {
          result[0] += 0.03236731;
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 66))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
              result[0] += 0.00022758842;
            } else {
              result[0] += 0.0045965556;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 40))) {
              result[0] += 0.0054974416;
            } else {
              result[0] += -0.00038611452;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 128))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 118))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 66))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 140))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
              result[0] += -0.030644778;
            } else {
              result[0] += -0.0012927776;
            }
          } else {
            result[0] += -0.027898073;
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 270))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 10))) {
              result[0] += -0.0026217196;
            } else {
              result[0] += 0.00031906084;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 14))) {
              result[0] += 0.053565513;
            } else {
              result[0] += 0.011517114;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 242))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 206))) {
              result[0] += 0.0022491028;
            } else {
              result[0] += 0.017666435;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 48))) {
              result[0] += -0.039555483;
            } else {
              result[0] += -0.007261213;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 24))) {
            result[0] += -0.0013352408;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 194))) {
              result[0] += 0.019903926;
            } else {
              result[0] += -0.0018211514;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 98))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 260))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
              result[0] += -0.003761566;
            } else {
              result[0] += 5.4693035e-05;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 44))) {
              result[0] += 0.066539966;
            } else {
              result[0] += 0.0024916714;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 282))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
              result[0] += 0.0024568306;
            } else {
              result[0] += 0.013490746;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 144))) {
              result[0] += 0.0017763426;
            } else {
              result[0] += -0.0017636567;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 66))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
            result[0] += -0.016284099;
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
              result[0] += 0.022716116;
            } else {
              result[0] += 0.0030027088;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 84))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 108))) {
              result[0] += -0.008329183;
            } else {
              result[0] += -0.02752207;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
              result[0] += 0.024196664;
            } else {
              result[0] += -0.0069028423;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 268))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 266))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 150))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 50))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 6))) {
              result[0] += 0.03250914;
            } else {
              result[0] += 0.0055702133;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
              result[0] += 0.008322871;
            } else {
              result[0] += -0.015406762;
            }
          }
        } else {
          result[0] += 0.0395282;
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 262))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 60))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 260))) {
              result[0] += -0.018292183;
            } else {
              result[0] += -0;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 294))) {
              result[0] += -0.0053772978;
            } else {
              result[0] += 0.0050149076;
            }
          }
        } else {
          result[0] += 0.006939339;
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 88))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 282))) {
          result[0] += 0.00481271;
        } else {
          result[0] += -0.0038877416;
        }
      } else {
        result[0] += -0.0105433585;
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 320))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 284))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 212))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 252))) {
              result[0] += 8.760832e-05;
            } else {
              result[0] += 0.0031985168;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 250))) {
              result[0] += -0.0029068196;
            } else {
              result[0] += 0.0005925631;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 242))) {
            result[0] += 0.019893859;
          } else {
            result[0] += 0.009958875;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 266))) {
          result[0] += -0.017648466;
        } else {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 304))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 302))) {
              result[0] += -0.011696982;
            } else {
              result[0] += 0.004698274;
            }
          } else {
            result[0] += -0.0020677263;
          }
        }
      }
    } else {
      result[0] += 0.04254042;
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 162))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 168))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 122))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 76))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 98))) {
              result[0] += -0.0029215065;
            } else {
              result[0] += 0.011221336;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 112))) {
              result[0] += 0.020535378;
            } else {
              result[0] += 0.0062516048;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 62))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 302))) {
              result[0] += -0.0063436194;
            } else {
              result[0] += -0.01713322;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 162))) {
              result[0] += 0.010334019;
            } else {
              result[0] += -0.0069331317;
            }
          }
        }
      } else {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 322))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 182))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 164))) {
              result[0] += -0.0077398308;
            } else {
              result[0] += -0.03391137;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 190))) {
              result[0] += 0.018701833;
            } else {
              result[0] += 0.005074762;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 192))) {
            result[0] += 0.05592671;
          } else {
            result[0] += 0.012150154;
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 138))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 44))) {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 84))) {
            result[0] += -0.004530545;
          } else {
            result[0] += -0.014109983;
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 34))) {
            result[0] += 0.016917644;
          } else {
            result[0] += -0.0003956896;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 168))) {
          result[0] += 0.018351296;
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 222))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 326))) {
              result[0] += 0.009391616;
            } else {
              result[0] += -0.012847346;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
              result[0] += 0.0028710503;
            } else {
              result[0] += -0.014146092;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 272))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 24))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
              result[0] += 0.004130573;
            } else {
              result[0] += -0.02545332;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 136))) {
              result[0] += -0.011545375;
            } else {
              result[0] += 0.0030030604;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 124))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 76))) {
              result[0] += 0.0053757923;
            } else {
              result[0] += 0.026795724;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += 0.0073435195;
            } else {
              result[0] += -0.011538415;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 84))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 14))) {
            result[0] += 0.010773177;
          } else {
            result[0] += -0.0014024094;
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 90))) {
            result[0] += -0.00650954;
          } else {
            result[0] += 0.017750045;
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 280))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 256))) {
            result[0] += -0.01760956;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 184))) {
              result[0] += -0.017441498;
            } else {
              result[0] += -0.0037122124;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 216))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 182))) {
              result[0] += -0.024908429;
            } else {
              result[0] += -0.012610437;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 240))) {
              result[0] += 0.020257859;
            } else {
              result[0] += -0.0005201062;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 342))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 20))) {
            result[0] += 0.031078367;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 338))) {
              result[0] += -0.03464001;
            } else {
              result[0] += 0.020424169;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 348))) {
            result[0] += 0.0028230688;
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 114))) {
              result[0] += -0.003942101;
            } else {
              result[0] += -0.017917613;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 56))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 12))) {
        result[0] += -0.018274682;
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 88))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
            result[0] += -0.014930627;
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 22))) {
              result[0] += 0.009653761;
            } else {
              result[0] += -0.005647427;
            }
          }
        } else {
          result[0] += -0.017578453;
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 216))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 134))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 116))) {
              result[0] += -0.004387402;
            } else {
              result[0] += 0.004307063;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 244))) {
              result[0] += 0.0050197043;
            } else {
              result[0] += -0.009600091;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 264))) {
              result[0] += -0.0004044637;
            } else {
              result[0] += 0.013372946;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 138))) {
              result[0] += -0.030934876;
            } else {
              result[0] += -0.005925291;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 6))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
              result[0] += -0.018131623;
            } else {
              result[0] += 0.007491207;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 158))) {
              result[0] += 0.00448369;
            } else {
              result[0] += -0.004200383;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 124))) {
              result[0] += 0.00013818998;
            } else {
              result[0] += -0.0060928585;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 226))) {
              result[0] += 0.003693558;
            } else {
              result[0] += -0.0001647349;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 400))) {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 388))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 346))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 356))) {
              result[0] += 0.00025481472;
            } else {
              result[0] += 0.015072713;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 56))) {
              result[0] += -0.0068883398;
            } else {
              result[0] += 0.005798895;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
              result[0] += -0.012064995;
            } else {
              result[0] += -0.00055150193;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 236))) {
              result[0] += 0.025693614;
            } else {
              result[0] += -0.0082374755;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 108))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 4))) {
              result[0] += 0.020123675;
            } else {
              result[0] += 0.00013938252;
            }
          } else {
            result[0] += -0.009953358;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 196))) {
            result[0] += 0.016889976;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 236))) {
              result[0] += -0.010803269;
            } else {
              result[0] += 0.010580909;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 392))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 198))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 154))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 144))) {
              result[0] += -0;
            } else {
              result[0] += 0.05305763;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 74))) {
              result[0] += -0.0071868575;
            } else {
              result[0] += 0.06390004;
            }
          }
        } else {
          result[0] += 0.057614993;
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 176))) {
          result[0] += -0.0067237015;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 398))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 52))) {
              result[0] += 0.0020582944;
            } else {
              result[0] += 0.009856465;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 86))) {
              result[0] += -0.0054150624;
            } else {
              result[0] += 0.0054157143;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 90))) {
      result[0] += -0.010108463;
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
        result[0] += 0.0012366887;
      } else {
        result[0] += -0.005106163;
      }
    }
  }
  if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 162))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 150))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 52))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.0030484337;
            } else {
              result[0] += 0.007103072;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 118))) {
              result[0] += -0.020119462;
            } else {
              result[0] += 0.0049716155;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 42))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 18))) {
              result[0] += -0.020776983;
            } else {
              result[0] += 0.0001524161;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
              result[0] += 0.009727257;
            } else {
              result[0] += 0.0006957262;
            }
          }
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 54))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 158))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 152))) {
              result[0] += -0.011365656;
            } else {
              result[0] += -0.0006019767;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 154))) {
              result[0] += 0.02111136;
            } else {
              result[0] += 0.001328925;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 216))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 298))) {
              result[0] += -0.009145065;
            } else {
              result[0] += 0.011142318;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 250))) {
              result[0] += -0.05126782;
            } else {
              result[0] += -0.02713782;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 138))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 268))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 66))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 42))) {
              result[0] += -0.003784838;
            } else {
              result[0] += 0.0068541127;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 30))) {
              result[0] += -0.029539952;
            } else {
              result[0] += -0.0067226845;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 166))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 44))) {
              result[0] += 0.0031095946;
            } else {
              result[0] += -0.031267606;
            }
          } else {
            result[0] += 0.03802244;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
          result[0] += -0.008942714;
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 332))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
              result[0] += 0.018265491;
            } else {
              result[0] += 0.01069651;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
              result[0] += 0.0040097563;
            } else {
              result[0] += -0.03305195;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 118))) {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 194))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 70))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 30))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 60))) {
              result[0] += -0.019308351;
            } else {
              result[0] += 0.0002994246;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
              result[0] += 0.0010901977;
            } else {
              result[0] += 0.0058678756;
            }
          }
        } else {
          result[0] += 0.024462478;
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 290))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 74))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += -0.012439872;
            } else {
              result[0] += 0.017688526;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 242))) {
              result[0] += -0.0040794015;
            } else {
              result[0] += -0.009616309;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 316))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 246))) {
              result[0] += 0.0013879368;
            } else {
              result[0] += 0.017187012;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 336))) {
              result[0] += -0.0060414565;
            } else {
              result[0] += 0.00086067413;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 210))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 288))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 150))) {
              result[0] += -0.022543944;
            } else {
              result[0] += 0.0002652502;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
              result[0] += 0.010252799;
            } else {
              result[0] += -0.0059008473;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 342))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 68))) {
              result[0] += -0.00482871;
            } else {
              result[0] += -0.036633775;
            }
          } else {
            result[0] += -0.020131879;
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 88))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
              result[0] += 0.019643709;
            } else {
              result[0] += -0.0055163004;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 110))) {
              result[0] += -0.023596639;
            } else {
              result[0] += -0.005768484;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 214))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 282))) {
              result[0] += 0.024731327;
            } else {
              result[0] += 0.0059899557;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 216))) {
              result[0] += -0.012860804;
            } else {
              result[0] += 0.00026836878;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 250))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 270))) {
      if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 216))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 4))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 188))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 20))) {
              result[0] += -0.015227991;
            } else {
              result[0] += 0.007824044;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 180))) {
              result[0] += -0.0030827313;
            } else {
              result[0] += 0.04585096;
            }
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 44))) {
              result[0] += 0.0038916196;
            } else {
              result[0] += -0.00018074422;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
              result[0] += -0;
            } else {
              result[0] += 0.01431708;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 8))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 40))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 254))) {
              result[0] += -0.03443868;
            } else {
              result[0] += -0.016926734;
            }
          } else {
            result[0] += 0.016495222;
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 318))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 150))) {
              result[0] += -0.0018124201;
            } else {
              result[0] += -0.008941532;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
              result[0] += 0.010480117;
            } else {
              result[0] += -0.0053374222;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 234))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 144))) {
          result[0] += -0.0025247212;
        } else {
          result[0] += -0.021172313;
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 278))) {
          result[0] += 0.0136889;
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 24))) {
              result[0] += -0;
            } else {
              result[0] += -0.028621003;
            }
          } else {
            result[0] += 0.008883018;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 272))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 178))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 200))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 120))) {
              result[0] += 0.004496847;
            } else {
              result[0] += -0.0019887805;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 60))) {
              result[0] += 0.011425511;
            } else {
              result[0] += 0.001471782;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 164))) {
              result[0] += 0.004373122;
            } else {
              result[0] += -0.0050846357;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 312))) {
              result[0] += -0.01642145;
            } else {
              result[0] += -0.0041971803;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 240))) {
            result[0] += -0.019498728;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 296))) {
              result[0] += 0.016818725;
            } else {
              result[0] += 0.031616382;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 256))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 152))) {
              result[0] += 0.0041652615;
            } else {
              result[0] += 0.024725467;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 4))) {
              result[0] += -0.017824838;
            } else {
              result[0] += 0.00069643237;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 104))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 354))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 346))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 138))) {
              result[0] += 0.057328846;
            } else {
              result[0] += -0.009282811;
            }
          } else {
            result[0] += -0.027812624;
          }
        } else {
          result[0] += 0.006232637;
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 208))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 184))) {
              result[0] += -0.0007241729;
            } else {
              result[0] += 0.01345086;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 372))) {
              result[0] += -0.01243884;
            } else {
              result[0] += -0.0010584429;
            }
          }
        } else {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 66))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 48))) {
              result[0] += 0.012493608;
            } else {
              result[0] += 0.0014906918;
            }
          } else {
            result[0] += -0.011661606;
          }
        }
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
      if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 110))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 2))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 216))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 58))) {
              result[0] += 0.0059683393;
            } else {
              result[0] += 0.047656022;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 76))) {
              result[0] += -0.01823411;
            } else {
              result[0] += 0.011243396;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 88))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 36))) {
              result[0] += -0.015964573;
            } else {
              result[0] += 0.018772287;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 84))) {
              result[0] += -0.024348408;
            } else {
              result[0] += 0.0071454565;
            }
          }
        }
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 60))) {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 10))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 118))) {
              result[0] += 0.0038073042;
            } else {
              result[0] += -0.010677494;
            }
          } else {
            result[0] += -0.010760325;
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 166))) {
            result[0] += -0.013037783;
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 196))) {
              result[0] += 0.04078368;
            } else {
              result[0] += -0;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 6))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 110))) {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 24))) {
            result[0] += -0.019799097;
          } else {
            result[0] += 0.0018352288;
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 14))) {
            result[0] += 0.016346006;
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 266))) {
              result[0] += -0.018813994;
            } else {
              result[0] += -0.007913149;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 174))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 296))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 280))) {
              result[0] += 0.000356507;
            } else {
              result[0] += 0.004809927;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 298))) {
              result[0] += -0.015981428;
            } else {
              result[0] += -0.0013808893;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 194))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 84))) {
              result[0] += -0.0013645035;
            } else {
              result[0] += -0.009478527;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 196))) {
              result[0] += 0.01796195;
            } else {
              result[0] += -0.00023798608;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 142))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 210))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 216))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 6))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 26))) {
              result[0] += -0.0045832377;
            } else {
              result[0] += -0.03190832;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
              result[0] += 0.0066647097;
            } else {
              result[0] += -0.002318192;
            }
          }
        } else {
          result[0] += -0.040041517;
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 264))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 270))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 54))) {
              result[0] += -0.0019740702;
            } else {
              result[0] += -0.024438191;
            }
          } else {
            result[0] += -0;
          }
        } else {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 166))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 2))) {
              result[0] += 0.029889846;
            } else {
              result[0] += 0.0023792638;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 268))) {
              result[0] += 0.0048653954;
            } else {
              result[0] += -0.013145282;
            }
          }
        }
      }
    } else {
      result[0] += 0.026439697;
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
    if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 50))) {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 92))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 154))) {
          result[0] += -0.023412313;
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 164))) {
            result[0] += 0.020196555;
          } else {
            result[0] += -0.012161604;
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 38))) {
          result[0] += 0.08170753;
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 94))) {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 18))) {
              result[0] += -0.002143487;
            } else {
              result[0] += -0.038541686;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 50))) {
              result[0] += -0.006611839;
            } else {
              result[0] += 0.008542123;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 200))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 132))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 110))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 46))) {
              result[0] += 0.0018188714;
            } else {
              result[0] += -0.015560075;
            }
          } else {
            result[0] += 0.021364069;
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 112))) {
            result[0] += -0.029760977;
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 148))) {
              result[0] += 0.0078332;
            } else {
              result[0] += -0.011854128;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 64))) {
          result[0] += 0.014748133;
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
            result[0] += -0.012884839;
          } else {
            result[0] += 0.0029220346;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 170))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 42))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 148))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
              result[0] += 0.015425849;
            } else {
              result[0] += -0.004185417;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 98))) {
              result[0] += -0.0011755283;
            } else {
              result[0] += -0.03457963;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 266))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 48))) {
              result[0] += -0.009746331;
            } else {
              result[0] += -0.030245284;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 290))) {
              result[0] += -0.006777007;
            } else {
              result[0] += -0.02345091;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 92))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
            result[0] += -0.008947103;
          } else {
            result[0] += 0.03920984;
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 148))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 6))) {
              result[0] += 0.009707036;
            } else {
              result[0] += -0.0060035028;
            }
          } else {
            result[0] += 0.01016276;
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 16))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 28))) {
          result[0] += -0;
        } else {
          result[0] += 0.032831904;
        }
      } else {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 120))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 118))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 98))) {
              result[0] += 6.658281e-06;
            } else {
              result[0] += 0.0031512368;
            }
          } else {
            result[0] += 0.02200143;
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
            result[0] += 0.033468585;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 82))) {
              result[0] += -0.008753168;
            } else {
              result[0] += -0.0016586205;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 314))) {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 18))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 26))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
              result[0] += -0.033794925;
            } else {
              result[0] += 0.0085956175;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 8))) {
              result[0] += 0.02062062;
            } else {
              result[0] += -0.0024983874;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 30))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
              result[0] += -0;
            } else {
              result[0] += 0.02111803;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 66))) {
              result[0] += 0.048572876;
            } else {
              result[0] += 0.007911704;
            }
          }
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 30))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 112))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 68))) {
              result[0] += -0.0077246563;
            } else {
              result[0] += 0.005386382;
            }
          } else {
            result[0] += -0.020841127;
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 48))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 4))) {
              result[0] += -0.014042695;
            } else {
              result[0] += 0.008498053;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 12))) {
              result[0] += -0.024558064;
            } else {
              result[0] += -0.0025908693;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 32))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 24))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 170))) {
              result[0] += 2.548086e-05;
            } else {
              result[0] += 0.026205242;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 92))) {
              result[0] += 0.009812611;
            } else {
              result[0] += 0.00306112;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
            result[0] += 0.027721763;
          } else {
            result[0] += 0.006687283;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 16))) {
            result[0] += 0.035541125;
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 18))) {
              result[0] += -0.017473036;
            } else {
              result[0] += -0.0030920196;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
              result[0] += -0.00029661623;
            } else {
              result[0] += -0.006844879;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.009076519;
            } else {
              result[0] += 0.00021811183;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 36))) {
      if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 34))) {
        result[0] += 0.026330112;
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 248))) {
          result[0] += 0.01319269;
        } else {
          result[0] += -0.004578778;
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 62))) {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
          result[0] += -0;
        } else {
          result[0] += -0.017771253;
        }
      } else {
        if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 68))) {
          result[0] += 0.041357838;
        } else {
          result[0] += 0.0004609677;
        }
      }
    }
  }
  if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 120))) {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 116))) {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 0))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 12))) {
          result[0] += -0.03449745;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 10))) {
              result[0] += -0.000830644;
            } else {
              result[0] += 0.027823929;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += -0.014675349;
            } else {
              result[0] += 0.002744519;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
              result[0] += 0.00021282148;
            } else {
              result[0] += 0.012012251;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
              result[0] += -0.023646448;
            } else {
              result[0] += -0.0012802199;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 64))) {
              result[0] += 0.005379937;
            } else {
              result[0] += 0.020074537;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
              result[0] += -0.017483069;
            } else {
              result[0] += 0.00045991183;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 28))) {
        if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 118))) {
          result[0] += -0.010277932;
        } else {
          result[0] += 0.020262284;
        }
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 78))) {
          result[0] += 0.026524289;
        } else {
          result[0] += 0.0005923277;
        }
      }
    }
  } else {
    if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 78))) {
      result[0] += -0.023216326;
    } else {
      if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 160))) {
        result[0] += -0.0063289674;
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 80))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 56))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 124))) {
              result[0] += -0.026424408;
            } else {
              result[0] += -0.00650081;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
              result[0] += 0.035873644;
            } else {
              result[0] += -0.0067583695;
            }
          }
        } else {
          if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 134))) {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 122))) {
              result[0] += 0.019181004;
            } else {
              result[0] += 0.0012452018;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.006571722;
            } else {
              result[0] += -0.00113232;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 248))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 270))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 184))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 104))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 312))) {
              result[0] += -0.001673722;
            } else {
              result[0] += 0.0015081331;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 18))) {
              result[0] += 0.019442735;
            } else {
              result[0] += -0.0041994313;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 116))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 170))) {
              result[0] += 0.00018035778;
            } else {
              result[0] += -0.012916463;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 356))) {
              result[0] += 0.003256688;
            } else {
              result[0] += -0.0061490457;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 230))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 28))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 80))) {
              result[0] += -0.002768964;
            } else {
              result[0] += 0.036568955;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
              result[0] += 0.0035748954;
            } else {
              result[0] += -0.0018361468;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 224))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
              result[0] += 0.0008316287;
            } else {
              result[0] += -0.012621154;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 212))) {
              result[0] += 0.0009690163;
            } else {
              result[0] += -0.009806303;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 16))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 226))) {
          result[0] += -0.008160834;
        } else {
          result[0] += 0.010674962;
        }
      } else {
        if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 66))) {
          result[0] += -0.018799648;
        } else {
          result[0] += -0.0010747229;
        }
      }
    }
  } else {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 274))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 252))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += 0.013668454;
            } else {
              result[0] += 0.0015821367;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 254))) {
              result[0] += 0.0013118708;
            } else {
              result[0] += -0.0027904937;
            }
          }
        } else {
          result[0] += 0.013429761;
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 150))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 312))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 278))) {
              result[0] += 0.0026331868;
            } else {
              result[0] += 0.02152044;
            }
          } else {
            result[0] += 0.010367058;
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 308))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
              result[0] += -0.009165699;
            } else {
              result[0] += 0.012586451;
            }
          } else {
            result[0] += 0.017478531;
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 130))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 354))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 56))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 138))) {
              result[0] += 0.03761896;
            } else {
              result[0] += -0.009919451;
            }
          } else {
            result[0] += -0.024488965;
          }
        } else {
          result[0] += 0.0063245073;
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 178))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 292))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 342))) {
              result[0] += 0.004199187;
            } else {
              result[0] += -0.0270401;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 12))) {
              result[0] += -0.034475163;
            } else {
              result[0] += 0.03449012;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 204))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 312))) {
              result[0] += -0.008276438;
            } else {
              result[0] += -0.021829268;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 348))) {
              result[0] += -0.00024369739;
            } else {
              result[0] += 0.018283892;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 0))) {
    if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 180))) {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 26))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.015541151;
            } else {
              result[0] += -0.004526817;
            }
          } else {
            result[0] += 0.04517481;
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 44))) {
              result[0] += -0.008011468;
            } else {
              result[0] += -0.0023013349;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 16))) {
              result[0] += 0.027930424;
            } else {
              result[0] += 0.001679267;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 92))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 20))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 32))) {
              result[0] += 0.022154698;
            } else {
              result[0] += -0.01756174;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 76))) {
              result[0] += -0.045003;
            } else {
              result[0] += -0.010276439;
            }
          }
        } else {
          result[0] += 0.001000655;
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 112))) {
        if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 116))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 26))) {
            result[0] += 0.0012434544;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 160))) {
              result[0] += 0.018014176;
            } else {
              result[0] += -0.00070313603;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 126))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 114))) {
              result[0] += 0.0062645376;
            } else {
              result[0] += -0.015609448;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 16))) {
              result[0] += 0.01479733;
            } else {
              result[0] += 0.0014734569;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 134))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 290))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 304))) {
              result[0] += -0;
            } else {
              result[0] += -0.012774549;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 40))) {
              result[0] += 0.014705256;
            } else {
              result[0] += -0.004232665;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 314))) {
              result[0] += 0.0029532865;
            } else {
              result[0] += -0.0029254516;
            }
          } else {
            result[0] += 0.026304556;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 128))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 256))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 150))) {
              result[0] += 0.00030722877;
            } else {
              result[0] += -0.0051551643;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
              result[0] += 0.016633255;
            } else {
              result[0] += 0.001668491;
            }
          }
        } else {
          result[0] += -0.014828484;
        }
      } else {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 236))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 254))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 286))) {
              result[0] += 0.0071044452;
            } else {
              result[0] += -0.0076894383;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 54))) {
              result[0] += 0.02084865;
            } else {
              result[0] += 0.007473866;
            }
          }
        } else {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 278))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 290))) {
              result[0] += 0.005741518;
            } else {
              result[0] += 0.03405652;
            }
          } else {
            result[0] += 0.025841344;
          }
        }
      }
    } else {
      if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 22))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 150))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 154))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 152))) {
              result[0] += 0.00015940488;
            } else {
              result[0] += 0.05232349;
            }
          } else {
            result[0] += -0.018910853;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 160))) {
            result[0] += -0.051273484;
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 106))) {
              result[0] += -0.02290248;
            } else {
              result[0] += -0.0016118204;
            }
          }
        }
      } else {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 64))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 192))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 162))) {
              result[0] += 0.0010482629;
            } else {
              result[0] += -0.016297817;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 196))) {
              result[0] += 0.013967775;
            } else {
              result[0] += 0.027914885;
            }
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 74))) {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 2))) {
              result[0] += 0.038794536;
            } else {
              result[0] += -0.0062946193;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
              result[0] += -2.64479e-05;
            } else {
              result[0] += 0.021996183;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 104))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 40))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 32))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 28))) {
              result[0] += -0.005473042;
            } else {
              result[0] += 0.011005775;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 136))) {
              result[0] += -0.011721667;
            } else {
              result[0] += 0.0078737205;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 46))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 38))) {
              result[0] += 0.0025443584;
            } else {
              result[0] += -0.016291086;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.00024633956;
            } else {
              result[0] += 0.027108392;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 26))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 10))) {
              result[0] += 0.03744457;
            } else {
              result[0] += 0.0014647435;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 136))) {
              result[0] += -0.0058734766;
            } else {
              result[0] += -0.02578007;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 50))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += -0.00944722;
            } else {
              result[0] += -0.03411841;
            }
          } else {
            result[0] += -0.001750114;
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 14))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 100))) {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 24))) {
              result[0] += 0.0006245327;
            } else {
              result[0] += 0.02487868;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 18))) {
              result[0] += 0.003864918;
            } else {
              result[0] += 0.04562244;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 114))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 94))) {
              result[0] += -0;
            } else {
              result[0] += -0.037629705;
            }
          } else {
            result[0] += 0.017797282;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 20))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
              result[0] += 0.03347585;
            } else {
              result[0] += -0.00015754919;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
              result[0] += 0.042547952;
            } else {
              result[0] += 0.008394706;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 34))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 304))) {
              result[0] += -0.0053476547;
            } else {
              result[0] += 0.019664986;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 274))) {
              result[0] += 0.00061302836;
            } else {
              result[0] += -0.008260283;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 112))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
        if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 164))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += -0.0029065635;
            } else {
              result[0] += 0.012601085;
            }
          } else {
            if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
              result[0] += 0.011005281;
            } else {
              result[0] += -0.010062504;
            }
          }
        } else {
          result[0] += 0.015878765;
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 202))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 182))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 154))) {
              result[0] += -0.012548672;
            } else {
              result[0] += -0.036353104;
            }
          } else {
            result[0] += -0.0037594133;
          }
        } else {
          result[0] += -0.026142243;
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 110))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 108))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 194))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 162))) {
              result[0] += 0.0049291016;
            } else {
              result[0] += 0.02395081;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 234))) {
              result[0] += -0.0030472444;
            } else {
              result[0] += 0.0040821205;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 100))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 114))) {
              result[0] += -0.020026503;
            } else {
              result[0] += -0.0037345327;
            }
          } else {
            result[0] += -0.025079226;
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 128))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 128))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 160))) {
              result[0] += 0.006035849;
            } else {
              result[0] += -0.005563633;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 134))) {
              result[0] += -0.017343946;
            } else {
              result[0] += 0.014413935;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 130))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 308))) {
              result[0] += 0.007634467;
            } else {
              result[0] += -0.024592249;
            }
          } else {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 106))) {
              result[0] += 0.0005188165;
            } else {
              result[0] += -0.006757953;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 298))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 52))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 240))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
              result[0] += 0.00393571;
            } else {
              result[0] += 0.02043746;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 262))) {
              result[0] += -0.01545644;
            } else {
              result[0] += -0.002095156;
            }
          }
        } else {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 210))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 128))) {
              result[0] += -0.00013221658;
            } else {
              result[0] += -0.002142299;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += -0.0012591215;
            } else {
              result[0] += 0.0015919211;
            }
          }
        }
      } else {
        result[0] += 0.011080801;
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 364))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 300))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 142))) {
            result[0] += -0.030891055;
          } else {
            result[0] += -0.011523842;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 296))) {
            result[0] += -0.0094786985;
          } else {
            result[0] += 0.00014606201;
          }
        }
      } else {
        result[0] += 0.045800555;
      }
    }
  } else {
    if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 18))) {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 0))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 6))) {
          result[0] += 0.021174032;
        } else {
          result[0] += -0;
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 54))) {
          result[0] += -0;
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 170))) {
            result[0] += -0.02313884;
          } else {
            result[0] += -0;
          }
        }
      }
    } else {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 310))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 120))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 170))) {
              result[0] += 0.0034402236;
            } else {
              result[0] += 0.03586924;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 150))) {
              result[0] += -0.007850896;
            } else {
              result[0] += 0.0023356122;
            }
          }
        } else {
          result[0] += 0.019080771;
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 340))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 216))) {
            result[0] += -0.006365792;
          } else {
            result[0] += 0.012724089;
          }
        } else {
          result[0] += -0.03172406;
        }
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 350))) {
    if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 144))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 142))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 116))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 138))) {
              result[0] += -0.00062813837;
            } else {
              result[0] += 0.002671611;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 86))) {
              result[0] += 0.00652712;
            } else {
              result[0] += 0.00037897687;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 152))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 286))) {
              result[0] += -0.00025623475;
            } else {
              result[0] += -0.004133305;
            }
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 82))) {
              result[0] += -0.01589259;
            } else {
              result[0] += 0.019172618;
            }
          }
        }
      } else {
        result[0] += 0.009757864;
      }
    } else {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 250))) {
        result[0] += -0.0065281326;
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 94))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 254))) {
            result[0] += 0.0007117824;
          } else {
            result[0] += 0.027270088;
          }
        } else {
          result[0] += -0.0023996648;
        }
      }
    }
  } else {
    if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 248))) {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 48))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
          result[0] += -0.0019857192;
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 222))) {
            result[0] += 0.025358735;
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 232))) {
              result[0] += -0.0021736256;
            } else {
              result[0] += 0.016332442;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 52))) {
          result[0] += -0.0129296705;
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 224))) {
            result[0] += -0;
          } else {
            result[0] += 0.018950501;
          }
        }
      }
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 68))) {
        result[0] += -0.0070409672;
      } else {
        result[0] += 0.010926886;
      }
    }
  }
  if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 314))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 312))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 306))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 248))) {
              result[0] += -0.00022246773;
            } else {
              result[0] += 0.00076635665;
            }
          } else {
            result[0] += 0.012082838;
          }
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 286))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 298))) {
              result[0] += -0.015522775;
            } else {
              result[0] += -0.0064152265;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 348))) {
              result[0] += -0.0013468252;
            } else {
              result[0] += 0.020515444;
            }
          }
        }
      } else {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 208))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 276))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 150))) {
              result[0] += 0.0077754804;
            } else {
              result[0] += -0.0030992497;
            }
          } else {
            result[0] += -0.017411573;
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 250))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
              result[0] += 0.003438398;
            } else {
              result[0] += 0.032958325;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 30))) {
              result[0] += 0.013451094;
            } else {
              result[0] += -0.006546424;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 32))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 188))) {
          result[0] += 0.009157052;
        } else {
          result[0] += -0.010829896;
        }
      } else {
        result[0] += 0.015605274;
      }
    }
  } else {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
      result[0] += 0.01869001;
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 228))) {
          result[0] += -0.0042858357;
        } else {
          result[0] += -0.02192712;
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 226))) {
          result[0] += -0.0021264374;
        } else {
          result[0] += 0.028851261;
        }
      }
    }
  }
  if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 150))) {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 6))) {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 282))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 2))) {
            result[0] += 0.008089608;
          } else {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 252))) {
              result[0] += -0.02760528;
            } else {
              result[0] += -0.014671764;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 248))) {
            result[0] += 0.0007742315;
          } else {
            result[0] += -0.014570135;
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 310))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 264))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 52))) {
              result[0] += 0.004098795;
            } else {
              result[0] += -0.0013311962;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 250))) {
              result[0] += -0.016194416;
            } else {
              result[0] += -0.004321107;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 246))) {
              result[0] += 0.012578972;
            } else {
              result[0] += 0.031848185;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 200))) {
              result[0] += 0.002893864;
            } else {
              result[0] += -0.0014323759;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 48))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 54))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 24))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += -0.0058673653;
            } else {
              result[0] += 0.014972388;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 32))) {
              result[0] += -0.016248314;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 42))) {
              result[0] += 0.001427743;
            } else {
              result[0] += 0.013427376;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 218))) {
              result[0] += 0.004056875;
            } else {
              result[0] += -0.0111101195;
            }
          }
        }
      } else {
        if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 142))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 108))) {
              result[0] += 0.009302656;
            } else {
              result[0] += 0.0009259971;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
              result[0] += -0.0071252473;
            } else {
              result[0] += 0.0005085042;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 118))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 100))) {
              result[0] += 0.0055279406;
            } else {
              result[0] += -0.015505721;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 152))) {
              result[0] += 0.022024345;
            } else {
              result[0] += 0.013201664;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 30))) {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 140))) {
        if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 24))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 154))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 34))) {
              result[0] += -0.0031792417;
            } else {
              result[0] += -0.024321148;
            }
          } else {
            result[0] += 0.0074777827;
          }
        } else {
          if (UNLIKELY((data[15].missing != -1) && (data[15].qvalue < 2))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 52))) {
              result[0] += -0.015280488;
            } else {
              result[0] += 0.004024551;
            }
          } else {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 26))) {
              result[0] += 0.013732604;
            } else {
              result[0] += -0;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 106))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 74))) {
            result[0] += -0.028851384;
          } else {
            result[0] += -0.051958527;
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 26))) {
            result[0] += 0.010219454;
          } else {
            result[0] += -0.009982391;
          }
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 162))) {
        if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 256))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 158))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 24))) {
              result[0] += 0.014080286;
            } else {
              result[0] += -0.004486435;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 136))) {
              result[0] += 0.005933493;
            } else {
              result[0] += 0.02153199;
            }
          }
        } else {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 316))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
              result[0] += -0.02032683;
            } else {
              result[0] += 0.002681443;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 214))) {
              result[0] += -0.0042459033;
            } else {
              result[0] += 0.0068887584;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 196))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 392))) {
              result[0] += 0.00047894806;
            } else {
              result[0] += -0.0061863232;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 230))) {
              result[0] += 0.008450966;
            } else {
              result[0] += 0.0015639787;
            }
          }
        } else {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 200))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 224))) {
              result[0] += -0.007651621;
            } else {
              result[0] += -0.0009728819;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 112))) {
              result[0] += 0.0062414994;
            } else {
              result[0] += -0.0019789643;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 314))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 312))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 306))) {
        if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 84))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 40))) {
              result[0] += -0.0028604511;
            } else {
              result[0] += 0.0004893718;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 224))) {
              result[0] += 0.026684532;
            } else {
              result[0] += 0.0063786306;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 16))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 206))) {
              result[0] += -0.009385272;
            } else {
              result[0] += 0.008824005;
            }
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 256))) {
              result[0] += -0.00080077257;
            } else {
              result[0] += 0.0010740731;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 374))) {
          if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 34))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 224))) {
              result[0] += 0.014482776;
            } else {
              result[0] += -0.0036570956;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 308))) {
              result[0] += 0.009366672;
            } else {
              result[0] += -0.00948626;
            }
          }
        } else {
          if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 26))) {
            result[0] += -0.005847118;
          } else {
            result[0] += 0.033575047;
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 218))) {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 188))) {
          result[0] += 0.0021236583;
        } else {
          result[0] += -0.01067119;
        }
      } else {
        result[0] += 0.0088947555;
      }
    }
  } else {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 222))) {
      result[0] += 0.018932056;
    } else {
      if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 64))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 230))) {
          result[0] += -0.0033845974;
        } else {
          result[0] += -0.020902513;
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 250))) {
          result[0] += 0.0032020702;
        } else {
          result[0] += 0.042684905;
        }
      }
    }
  }
  if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 18))) {
    if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 70))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 68))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 6))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 10))) {
              result[0] += -0.008612228;
            } else {
              result[0] += 0.0040322477;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 22))) {
              result[0] += -0.008511812;
            } else {
              result[0] += 0.011585411;
            }
          }
        } else {
          if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 18))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 44))) {
              result[0] += 1.8034565e-05;
            } else {
              result[0] += 0.027274786;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 64))) {
              result[0] += 0.008447873;
            } else {
              result[0] += -0.025014887;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 130))) {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 0))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 58))) {
              result[0] += -0.007062527;
            } else {
              result[0] += 0.014613422;
            }
          } else {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 24))) {
              result[0] += 0.0011185434;
            } else {
              result[0] += 0.008517242;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
            result[0] += -0.025690308;
          } else {
            result[0] += -0.0029228963;
          }
        }
      }
    } else {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 120))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 30))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 46))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
              result[0] += -0.017073292;
            } else {
              result[0] += 0.0035216322;
            }
          } else {
            result[0] += -0.023047855;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 22))) {
              result[0] += -0.025554648;
            } else {
              result[0] += -0.0017376606;
            }
          } else {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 106))) {
              result[0] += -0.0054964907;
            } else {
              result[0] += 0.037392512;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
          result[0] += -0.05761969;
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 34))) {
            if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 98))) {
              result[0] += -0.016015155;
            } else {
              result[0] += 0.0068704;
            }
          } else {
            if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 76))) {
              result[0] += -0.015397488;
            } else {
              result[0] += -0.04052282;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 138))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 370))) {
        if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 86))) {
          if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 4))) {
            if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 52))) {
              result[0] += 0.0003938696;
            } else {
              result[0] += 0.0045072534;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 64))) {
              result[0] += -0.0024714952;
            } else {
              result[0] += 0.0049051414;
            }
          }
        } else {
          result[0] += 0.017899476;
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 380))) {
          result[0] += -0.008237792;
        } else {
          result[0] += -0.0031683035;
        }
      }
    } else {
      if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 288))) {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 282))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 174))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 100))) {
              result[0] += -0.0050061867;
            } else {
              result[0] += 0.00018127509;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 26))) {
              result[0] += -0.0051624607;
            } else {
              result[0] += 0.00025673906;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 156))) {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 240))) {
              result[0] += -0.0046179085;
            } else {
              result[0] += 0.026366374;
            }
          } else {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 8))) {
              result[0] += -0;
            } else {
              result[0] += 0.018350547;
            }
          }
        }
      } else {
        if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 268))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 50))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 220))) {
              result[0] += -0.0091535505;
            } else {
              result[0] += -0.042126443;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 248))) {
              result[0] += -0.03904317;
            } else {
              result[0] += -0.0013817366;
            }
          }
        } else {
          if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 44))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 122))) {
              result[0] += 0.0032674843;
            } else {
              result[0] += 0.038644753;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
              result[0] += -0.003548465;
            } else {
              result[0] += 0.044352487;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 168))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
      if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 102))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 32))) {
              result[0] += -0.0006659017;
            } else {
              result[0] += -0.017484738;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 124))) {
              result[0] += 0.021091629;
            } else {
              result[0] += -0.0017686317;
            }
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 0))) {
            result[0] += -0.035111748;
          } else {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 46))) {
              result[0] += -0.0162628;
            } else {
              result[0] += -0.004374349;
            }
          }
        }
      } else {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 8))) {
          result[0] += -0.031120077;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
              result[0] += -0.0055634947;
            } else {
              result[0] += -0.017814448;
            }
          } else {
            if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 120))) {
              result[0] += 0.042426206;
            } else {
              result[0] += -0.0013250493;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
        if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 58))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 132))) {
            result[0] += 0.03963465;
          } else {
            result[0] += 0.018596584;
          }
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 114))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 94))) {
              result[0] += -0.0008957187;
            } else {
              result[0] += -0.03431357;
            }
          } else {
            result[0] += 0.015205196;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 52))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 84))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 102))) {
              result[0] += 0.012878765;
            } else {
              result[0] += -0.013091402;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 140))) {
              result[0] += -0.022308733;
            } else {
              result[0] += 0.0014070682;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 38))) {
              result[0] += -0.018751746;
            } else {
              result[0] += -0.041268926;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 52))) {
              result[0] += -0.0035958923;
            } else {
              result[0] += 9.794186e-05;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 352))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 198))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 120))) {
            result[0] += 0.013142203;
          } else {
            result[0] += -0.0014631682;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
            result[0] += -0;
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 16))) {
              result[0] += -0.01765822;
            } else {
              result[0] += 0.0049266764;
            }
          }
        }
      } else {
        result[0] += -0.0039253565;
      }
    } else {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 368))) {
        result[0] += 0.015282332;
      } else {
        result[0] += -0.0067160674;
      }
    }
  }
  if (LIKELY((data[10].missing != -1) && (data[10].qvalue < 84))) {
    if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 214))) {
      if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 176))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
          if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 160))) {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 32))) {
              result[0] += -0.0012096122;
            } else {
              result[0] += 0.004681909;
            }
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 118))) {
              result[0] += -0.013531153;
            } else {
              result[0] += 0.004223753;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
            if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 72))) {
              result[0] += -0.0026232149;
            } else {
              result[0] += -0.01515016;
            }
          } else {
            if (LIKELY((data[11].missing != -1) && (data[11].qvalue < 170))) {
              result[0] += -0.00014745975;
            } else {
              result[0] += 0.0059427386;
            }
          }
        }
      } else {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 86))) {
          if (LIKELY((data[17].missing != -1) && (data[17].qvalue < 322))) {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 146))) {
              result[0] += 0.011954216;
            } else {
              result[0] += -0.00069962896;
            }
          } else {
            result[0] += 0.03788197;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += -0.0036799812;
            } else {
              result[0] += -0.014614584;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 194))) {
              result[0] += -0;
            } else {
              result[0] += 0.020192096;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 246))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 96))) {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 218))) {
            if (UNLIKELY((data[10].missing != -1) && (data[10].qvalue < 36))) {
              result[0] += 0.010108634;
            } else {
              result[0] += 0.0014457417;
            }
          } else {
            if (UNLIKELY((data[12].missing != -1) && (data[12].qvalue < 138))) {
              result[0] += -0.0062636524;
            } else {
              result[0] += 0.00020081793;
            }
          }
        } else {
          if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 6))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 74))) {
              result[0] += -0.011251283;
            } else {
              result[0] += 0.0056470656;
            }
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 232))) {
              result[0] += 0.0014956547;
            } else {
              result[0] += 0.0059374445;
            }
          }
        }
      } else {
        if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 266))) {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 58))) {
            if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 114))) {
              result[0] += -0.010740235;
            } else {
              result[0] += 0.00111787;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += -0.001297196;
            } else {
              result[0] += 0.0012389937;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 240))) {
              result[0] += 0.01125375;
            } else {
              result[0] += -0.03845505;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 292))) {
              result[0] += -0.008091416;
            } else {
              result[0] += 0.0020921114;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
      if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 30))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
          if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 8))) {
            result[0] += -0.040072184;
          } else {
            if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 136))) {
              result[0] += 0.013737966;
            } else {
              result[0] += -0.00681287;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 132))) {
            if (LIKELY((data[12].missing != -1) && (data[12].qvalue < 0))) {
              result[0] += 0.028193597;
            } else {
              result[0] += -0.0037449375;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
              result[0] += 0.015007503;
            } else {
              result[0] += -0.009241991;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
          if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 94))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 18))) {
              result[0] += -0.003027987;
            } else {
              result[0] += -0.019359348;
            }
          } else {
            result[0] += 0.0025615941;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
            result[0] += 0.023024825;
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 218))) {
              result[0] += -0.0007633574;
            } else {
              result[0] += -0.006595341;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[15].missing != -1) && (data[15].qvalue < 6))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 146))) {
          result[0] += -0.039751314;
        } else {
          if (UNLIKELY((data[11].missing != -1) && (data[11].qvalue < 108))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 308))) {
              result[0] += 0.011828893;
            } else {
              result[0] += -0.009816745;
            }
          } else {
            result[0] += -0.015453085;
          }
        }
      } else {
        result[0] += 0.0023617218;
      }
    }
  }
  if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 322))) {
    if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 282))) {
      if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 276))) {
        if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 274))) {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 272))) {
            if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 268))) {
              result[0] += -2.5651694e-05;
            } else {
              result[0] += 0.004297465;
            }
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 234))) {
              result[0] += -0.012060625;
            } else {
              result[0] += 0.027334878;
            }
          }
        } else {
          result[0] += 0.0077310125;
        }
      } else {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 112))) {
          result[0] += -0.012176646;
        } else {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 306))) {
            result[0] += 0.0043146396;
          } else {
            if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 278))) {
              result[0] += -0.010458501;
            } else {
              result[0] += -0.0025602286;
            }
          }
        }
      }
    } else {
      result[0] += 0.026633803;
    }
  } else {
    if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 60))) {
      if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 130))) {
        if (UNLIKELY((data[14].missing != -1) && (data[14].qvalue < 14))) {
          if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 122))) {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 106))) {
              result[0] += 0.009240702;
            } else {
              result[0] += 0.02637141;
            }
          } else {
            result[0] += -0.0056706998;
          }
        } else {
          if (LIKELY((data[13].missing != -1) && (data[13].qvalue < 10))) {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 286))) {
              result[0] += -0.003625507;
            } else {
              result[0] += 0.0070409435;
            }
          } else {
            if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 280))) {
              result[0] += 0.014894609;
            } else {
              result[0] += -0.003168697;
            }
          }
        }
      } else {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 178))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
            result[0] += -0.015371713;
          } else {
            if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 148))) {
              result[0] += -0.009225684;
            } else {
              result[0] += 0.00080344983;
            }
          }
        } else {
          result[0] += -0.034244932;
        }
      }
    } else {
      if (UNLIKELY((data[13].missing != -1) && (data[13].qvalue < 72))) {
        if (LIKELY((data[9].missing != -1) && (data[9].qvalue < 208))) {
          if (UNLIKELY((data[9].missing != -1) && (data[9].qvalue < 184))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += -0.013472222;
            } else {
              result[0] += 0.00748849;
            }
          } else {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 304))) {
              result[0] += 0.0054984465;
            } else {
              result[0] += 0.048147988;
            }
          }
        } else {
          result[0] += -0.010059855;
        }
      } else {
        if (LIKELY((data[16].missing != -1) && (data[16].qvalue < 346))) {
          if (LIKELY((data[14].missing != -1) && (data[14].qvalue < 250))) {
            if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 278))) {
              result[0] += 0.0179956;
            } else {
              result[0] += 0.00028234924;
            }
          } else {
            if (UNLIKELY((data[16].missing != -1) && (data[16].qvalue < 330))) {
              result[0] += -0.005519641;
            } else {
              result[0] += -0.03736005;
            }
          }
        } else {
          if (UNLIKELY((data[17].missing != -1) && (data[17].qvalue < 226))) {
            result[0] += -0.0048794984;
          } else {
            result[0] += 0.007994923;
          }
        }
      }
    }
  }

  // Apply base_scores
  result[0] += -1.882233619689941406;
  result[0] = std::exp(result[0]);

  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void dualsimplex_predictor::postprocess(double* result)
{
  // Do nothing
}

// Feature names array
const char* dualsimplex_predictor::feature_names[dualsimplex_predictor::NUM_FEATURES] = {
  "m",
  "n",
  "nnz",
  "density",
  "avg_nnz_col",
  "avg_nnz_row",
  "bounded",
  "free",
  "refact_freq",
  "num_refacts",
  "num_updates",
  "sparse_dz",
  "dense_dz",
  "bound_flips",
  "num_infeas",
  "dy_nz_pct",
  "byte_loads",
  "byte_stores"};
