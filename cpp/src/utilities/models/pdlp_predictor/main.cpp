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
};
static const int32_t num_class[] = {
  1,
};

int32_t pdlp_predictor::get_num_target(void) { return N_TARGET; }
void pdlp_predictor::get_num_class(int32_t* out)
{
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t pdlp_predictor::get_num_feature(void) { return 8; }
const char* pdlp_predictor::get_threshold_type(void) { return "float32"; }
const char* pdlp_predictor::get_leaf_output_type(void) { return "float32"; }

void pdlp_predictor::predict(union Entry* data, int pred_margin, double* result)
{
  // Quantize data
  for (int i = 0; i < 8; ++i) {
    if (data[i].missing != -1 && !is_categorical[i]) {
      data[i].qvalue = quantize(data[i].fvalue, i);
    }
  }

  unsigned int tmp;
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 14))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 30))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
            result[0] += -0.014820111;
          } else {
            result[0] += 0.0105687855;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
              result[0] += -0.053651925;
            } else {
              result[0] += -0.10176092;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 220))) {
              result[0] += -0.012739944;
            } else {
              result[0] += -0.036262058;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
          result[0] += -0.0063560107;
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 258))) {
            result[0] += 0.08327574;
          } else {
            result[0] += 0.026058618;
          }
        }
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 230))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 164))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
              result[0] += 0.031961214;
            } else {
              result[0] += 0.04515064;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
              result[0] += 0.018566301;
            } else {
              result[0] += -0.009213644;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
              result[0] += 0.14206316;
            } else {
              result[0] += 0.07914438;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
              result[0] += 0.025240844;
            } else {
              result[0] += 0.058222026;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.26381668;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 272))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
              result[0] += 0.085295476;
            } else {
              result[0] += 0.16802387;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 258))) {
              result[0] += 0.19548021;
            } else {
              result[0] += 0.14702512;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0039215386;
            } else {
              result[0] += -0.020411594;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
              result[0] += -0.030691355;
            } else {
              result[0] += -0.066723615;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 182))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 146))) {
              result[0] += -0.039679535;
            } else {
              result[0] += -0.07033275;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 226))) {
              result[0] += -0.017522344;
            } else {
              result[0] += 0.024236064;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 76))) {
              result[0] += -0.01896934;
            } else {
              result[0] += 0.005805307;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.05819216;
            } else {
              result[0] += -0.030700704;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 210))) {
              result[0] += -0.016364044;
            } else {
              result[0] += -0.0079360735;
            }
          } else {
            result[0] += 0.09915119;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 260))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
              result[0] += 0.0082174195;
            } else {
              result[0] += -0.0037611835;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
              result[0] += 0.05758765;
            } else {
              result[0] += 0.00845852;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 188))) {
            result[0] += 0.08059614;
          } else {
            result[0] += 0.060384076;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 62))) {
              result[0] += -0;
            } else {
              result[0] += 0.07759698;
            }
          } else {
            result[0] += 0.12749861;
          }
        } else {
          result[0] += 0.16187607;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 14))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 22))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
              result[0] += -0.0052639237;
            } else {
              result[0] += 0.022940176;
            }
          } else {
            result[0] += -0.03480838;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 28))) {
              result[0] += -0.04990984;
            } else {
              result[0] += -0.080560416;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += -0.014295327;
            } else {
              result[0] += -0.03235691;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
          result[0] += -0;
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 258))) {
            result[0] += 0.07372898;
          } else {
            result[0] += 0.024104217;
          }
        }
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 230))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 176))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 232))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
              result[0] += 0.02912726;
            } else {
              result[0] += 0.04095058;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += 0.016825125;
            } else {
              result[0] += -0.007768286;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
              result[0] += 0.1265556;
            } else {
              result[0] += 0.07124209;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
              result[0] += 0.040830098;
            } else {
              result[0] += 0.0559672;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.23753951;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 272))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
              result[0] += 0.07685151;
            } else {
              result[0] += 0.1544579;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 260))) {
              result[0] += 0.046362683;
            } else {
              result[0] += 0.17204122;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += -0.022617795;
            } else {
              result[0] += -0.0122455275;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
              result[0] += -0.027298182;
            } else {
              result[0] += -0.059523176;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 182))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 248))) {
              result[0] += -0.035259727;
            } else {
              result[0] += -0.061327398;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 236))) {
              result[0] += -0.01425799;
            } else {
              result[0] += 0.017331526;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
              result[0] += 0.00193814;
            } else {
              result[0] += -0.017574918;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 196))) {
              result[0] += -0.02869076;
            } else {
              result[0] += -0.065549426;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
              result[0] += 0.0057360423;
            } else {
              result[0] += -0.013489055;
            }
          } else {
            result[0] += 0.089799576;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 260))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
              result[0] += 0.007112731;
            } else {
              result[0] += -0.0027204938;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
              result[0] += 0.054216303;
            } else {
              result[0] += 0.007258077;
            }
          }
        } else {
          result[0] += 0.059579976;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 152))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 62))) {
              result[0] += 0.0030776516;
            } else {
              result[0] += 0.05093806;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
              result[0] += 0.07593134;
            } else {
              result[0] += 0.11205458;
            }
          }
        } else {
          result[0] += 0.14210285;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 14))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 32))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
              result[0] += -0.014897979;
            } else {
              result[0] += 0.003548234;
            }
          } else {
            result[0] += 0.018050188;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 88))) {
              result[0] += -0.008003016;
            } else {
              result[0] += -0.029365538;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
              result[0] += -0.008477488;
            } else {
              result[0] += -0.05125452;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
            result[0] += 0.02146909;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 8))) {
              result[0] += -0.038308386;
            } else {
              result[0] += 0.0050441376;
            }
          }
        } else {
          result[0] += 0.045694016;
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
              result[0] += 0.010466478;
            } else {
              result[0] += 0.032060686;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.055632647;
            } else {
              result[0] += 0.10817957;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 244))) {
              result[0] += 0.05290089;
            } else {
              result[0] += 0.15234004;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += 0.11994175;
            } else {
              result[0] += 0.16568963;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
          result[0] += 0.21408843;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
            result[0] += 0.109505884;
          } else {
            result[0] += 0.16057345;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += -0.021672187;
            } else {
              result[0] += -0.05343589;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0054487307;
            } else {
              result[0] += -0.017187914;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 92))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 188))) {
              result[0] += -0.019816956;
            } else {
              result[0] += -0.0050099557;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 248))) {
              result[0] += -0.03200326;
            } else {
              result[0] += -0.056725156;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
              result[0] += -0.011511999;
            } else {
              result[0] += -0.028344637;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.024851266;
            } else {
              result[0] += -0.05057622;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 18))) {
              result[0] += -0.0015090223;
            } else {
              result[0] += 0.04174643;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 206))) {
              result[0] += -0.014773312;
            } else {
              result[0] += -0.001539268;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 210))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
              result[0] += 0.007884377;
            } else {
              result[0] += 0.061007004;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
              result[0] += 0.021275358;
            } else {
              result[0] += 0.004744153;
            }
          }
        } else {
          result[0] += 0.053870168;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += 0.00918735;
            } else {
              result[0] += 0.06173278;
            }
          } else {
            result[0] += 0.098605655;
          }
        } else {
          result[0] += 0.13258949;
        }
      }
    }
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 266))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 214))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
              result[0] += 0.0034634469;
            } else {
              result[0] += -0.01012047;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 202))) {
              result[0] += 0.026168926;
            } else {
              result[0] += -0.003920808;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 34))) {
              result[0] += 0.01599789;
            } else {
              result[0] += -0.02167696;
            }
          } else {
            result[0] += -0.057240784;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 180))) {
              result[0] += 0.016893527;
            } else {
              result[0] += -0.0182766;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
              result[0] += -0.027289895;
            } else {
              result[0] += -0.0045931367;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 236))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
              result[0] += 0.06542163;
            } else {
              result[0] += 0.048862282;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 248))) {
              result[0] += 0.012741444;
            } else {
              result[0] += -0.0075932243;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 250))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 64))) {
            result[0] += 0.090353966;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 260))) {
              result[0] += 0.042119607;
            } else {
              result[0] += 0.053020816;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 46))) {
              result[0] += 0.03130157;
            } else {
              result[0] += 0.007205482;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
              result[0] += 0.10881876;
            } else {
              result[0] += 0.054842055;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += 0.031206427;
            } else {
              result[0] += 0.093665555;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
              result[0] += 0.058885314;
            } else {
              result[0] += 0.009160067;
            }
          }
        } else {
          result[0] += 0.12904838;
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 196))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 132))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 126))) {
              result[0] += 0.009280808;
            } else {
              result[0] += 0.039755784;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 184))) {
              result[0] += -0.0042881714;
            } else {
              result[0] += 0.02849459;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 176))) {
              result[0] += 0.013534146;
            } else {
              result[0] += 0.042829912;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 196))) {
              result[0] += 0.060934413;
            } else {
              result[0] += 0.04222716;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 166))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 252))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 242))) {
              result[0] += -0.014791851;
            } else {
              result[0] += 0.03050575;
            }
          } else {
            result[0] += -0.051454216;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 244))) {
              result[0] += -0.00703993;
            } else {
              result[0] += 0.006275458;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 230))) {
              result[0] += 0.05098787;
            } else {
              result[0] += 0.0045872494;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 236))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 156))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 204))) {
              result[0] += 0.010856108;
            } else {
              result[0] += -0.028078709;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 196))) {
              result[0] += 0.059345823;
            } else {
              result[0] += 0.025444312;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 246))) {
            result[0] += 0.1018459;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 226))) {
              result[0] += 0.06307183;
            } else {
              result[0] += 0.049246445;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 192))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 248))) {
              result[0] += -0.0057842466;
            } else {
              result[0] += 0.03709874;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 272))) {
              result[0] += 0.10474528;
            } else {
              result[0] += 0.0065343017;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
              result[0] += 0.14898759;
            } else {
              result[0] += 0.09902468;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 204))) {
              result[0] += -0.020444617;
            } else {
              result[0] += 0.08464522;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 198))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += -0.010948095;
            } else {
              result[0] += -0.023803001;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += -0.011822949;
            } else {
              result[0] += 0.045119066;
            }
          }
        } else {
          result[0] += -0.06180344;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
          result[0] += -0.005668474;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 258))) {
            result[0] += 0.06234616;
          } else {
            result[0] += 0.017185496;
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 178))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 100))) {
              result[0] += 0.024365185;
            } else {
              result[0] += 0.034411617;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
              result[0] += 0.0037991663;
            } else {
              result[0] += 0.06540612;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 226))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
              result[0] += 0.04412658;
            } else {
              result[0] += 0.09097751;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
              result[0] += 0.109255195;
            } else {
              result[0] += 0.15402773;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
          result[0] += 0.18228906;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 266))) {
            result[0] += 0.091413066;
          } else {
            result[0] += 0.12995003;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 66))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
              result[0] += -0.013535394;
            } else {
              result[0] += -0.030926574;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
              result[0] += -0.02790393;
            } else {
              result[0] += -0.0042497306;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
              result[0] += -0.016862048;
            } else {
              result[0] += -0.042121567;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += 0.043057855;
            } else {
              result[0] += -0.010520599;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 268))) {
              result[0] += 0.0006321628;
            } else {
              result[0] += 0.043750294;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
              result[0] += 0.017152889;
            } else {
              result[0] += 0.0047487523;
            }
          }
        } else {
          result[0] += 0.047594197;
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
              result[0] += -0.0008511189;
            } else {
              result[0] += -0.015743343;
            }
          } else {
            result[0] += 0.020423314;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 252))) {
            result[0] += 0.071650326;
          } else {
            result[0] += 0.038795434;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
          result[0] += 0.106261484;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
            result[0] += 0.05060423;
          } else {
            result[0] += 0.09152374;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 32))) {
            result[0] += -0.0061984994;
          } else {
            result[0] += 0.037181865;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
            result[0] += 0.05393646;
          } else {
            result[0] += 0.015753375;
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 134))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
              result[0] += -0.021209892;
            } else {
              result[0] += 0.0069300993;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
              result[0] += -0.0077880546;
            } else {
              result[0] += -0.04197134;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 12))) {
            result[0] += 0.040751807;
          } else {
            result[0] += -0.002328172;
          }
        }
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 254))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 198))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
              result[0] += 0.011177325;
            } else {
              result[0] += -0.0057386267;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 100))) {
              result[0] += 0.022035722;
            } else {
              result[0] += 0.031592578;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 154))) {
              result[0] += 0.046259467;
            } else {
              result[0] += 0.07964405;
            }
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.037900005;
            } else {
              result[0] += 0.006691271;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 228))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.045320172;
            } else {
              result[0] += 0.08683389;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.112302594;
            } else {
              result[0] += 0.15148906;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 218))) {
            result[0] += 0.09804384;
          } else {
            result[0] += 0.15994717;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += -0.014513046;
            } else {
              result[0] += -0.04194841;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 60))) {
              result[0] += -0.018427247;
            } else {
              result[0] += -0.009509578;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 182))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 64))) {
              result[0] += -0.021712733;
            } else {
              result[0] += -0.033654116;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
              result[0] += 0.017070105;
            } else {
              result[0] += -0.010606452;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
              result[0] += 0.003849056;
            } else {
              result[0] += -0.012776096;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += -0.01715595;
            } else {
              result[0] += -0.034953322;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
              result[0] += 0.004928622;
            } else {
              result[0] += -0.009454594;
            }
          } else {
            result[0] += 0.06586691;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += 0.00017961742;
            } else {
              result[0] += -0.0068662055;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.034061044;
            } else {
              result[0] += 0.004285582;
            }
          }
        } else {
          result[0] += 0.041407876;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
              result[0] += 0.0008129136;
            } else {
              result[0] += 0.046276525;
            }
          } else {
            result[0] += 0.08599146;
          }
        } else {
          result[0] += 0.092344016;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 198))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 22))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 6))) {
              result[0] += 0.0022713204;
            } else {
              result[0] += -0.01581463;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
              result[0] += -0.019155947;
            } else {
              result[0] += -0.031192351;
            }
          }
        } else {
          result[0] += -0.054840814;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
            result[0] += 0.034809362;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.03447778;
            } else {
              result[0] += -0.0010464619;
            }
          }
        } else {
          result[0] += 0.036184475;
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 178))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
              result[0] += 0.011472473;
            } else {
              result[0] += -0.00536993;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
              result[0] += 0.017885301;
            } else {
              result[0] += 0.026684735;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 226))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 36))) {
              result[0] += 0.050483853;
            } else {
              result[0] += 0.032498028;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
              result[0] += 0.107258014;
            } else {
              result[0] += 0.05922347;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
          result[0] += 0.14826694;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 266))) {
            result[0] += 0.07493075;
          } else {
            result[0] += 0.105560794;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 202))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
              result[0] += -0.024180638;
            } else {
              result[0] += -0.010499556;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 218))) {
              result[0] += 0.016562223;
            } else {
              result[0] += -0.013134924;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
              result[0] += -0.018523889;
            } else {
              result[0] += -0.030611722;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += -0.017535986;
            } else {
              result[0] += -0.004351136;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 146))) {
              result[0] += -0.021926327;
            } else {
              result[0] += -0.008350995;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 42))) {
              result[0] += -0.011140397;
            } else {
              result[0] += -0.04280148;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += 0.03963171;
            } else {
              result[0] += 0.019383498;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 214))) {
              result[0] += -0.00729556;
            } else {
              result[0] += -0.040553037;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
          result[0] += 0.08876854;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += 0.067758165;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
              result[0] += 0.03981258;
            } else {
              result[0] += 0.05752442;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 226))) {
              result[0] += -0.00089354016;
            } else {
              result[0] += 0.0065446827;
            }
          } else {
            result[0] += -0.014053066;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 222))) {
            result[0] += 0.06474081;
          } else {
            result[0] += 0.017832294;
          }
        }
      }
    }
  }
  if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 206))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 202))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 100))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 30))) {
              result[0] += 0.0055920025;
            } else {
              result[0] += -0.013393702;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
              result[0] += 0.03213504;
            } else {
              result[0] += 0.004373153;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 204))) {
              result[0] += -0.026502002;
            } else {
              result[0] += -0.0089536635;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 108))) {
              result[0] += 0.023841746;
            } else {
              result[0] += -0.009630135;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 132))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 116))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 84))) {
              result[0] += 0.03338425;
            } else {
              result[0] += -0.007993841;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 98))) {
              result[0] += 0.039039984;
            } else {
              result[0] += 0.017829265;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 122))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
              result[0] += -0.008180471;
            } else {
              result[0] += -0.017685762;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 172))) {
              result[0] += 0.024588756;
            } else {
              result[0] += 0.00014103504;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 120))) {
        result[0] += -0.017513642;
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 228))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 204))) {
            result[0] += 0.053347778;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
              result[0] += 0.036463372;
            } else {
              result[0] += 0.019543491;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += 0.06796604;
            } else {
              result[0] += 0.011818393;
            }
          } else {
            result[0] += -0.003918262;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 260))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 252))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 240))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 224))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 108))) {
              result[0] += 0.055681136;
            } else {
              result[0] += 0.03390036;
            }
          } else {
            result[0] += 0.014793222;
          }
        } else {
          result[0] += 0.06464329;
        }
      } else {
        result[0] += 0.1336614;
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 214))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 200))) {
              result[0] += -0.010694483;
            } else {
              result[0] += 0.0089774085;
            }
          } else {
            result[0] += 0.09665075;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 214))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
              result[0] += 0.005944577;
            } else {
              result[0] += 0.039748937;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
              result[0] += -0.036054015;
            } else {
              result[0] += -0.03007635;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
              result[0] += 0.07873906;
            } else {
              result[0] += 0.040747557;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 280))) {
              result[0] += 0.10715278;
            } else {
              result[0] += 0.07093804;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 272))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 228))) {
              result[0] += 0.0400591;
            } else {
              result[0] += 0.070340686;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 226))) {
              result[0] += 0.006239193;
            } else {
              result[0] += 0.04050245;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 20))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
            result[0] += 0.0143945515;
          } else {
            result[0] += -0.01724617;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 124))) {
            result[0] += 0.06920057;
          } else {
            result[0] += 0.012906248;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
            result[0] += -0.028011957;
          } else {
            result[0] += -0.054441918;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 222))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += -0;
            } else {
              result[0] += -0.015584071;
            }
          } else {
            result[0] += 0.04294205;
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 176))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
              result[0] += 0.0010215238;
            } else {
              result[0] += 0.018780788;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 36))) {
              result[0] += 0.038763847;
            } else {
              result[0] += 0.023370415;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 224))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.032697503;
            } else {
              result[0] += 0.13195097;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += 0.074951015;
            } else {
              result[0] += 0.10605689;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
          result[0] += 0.12042389;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 6))) {
              result[0] += 0.07597848;
            } else {
              result[0] += 0.059126675;
            }
          } else {
            result[0] += 0.087247975;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 212))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
              result[0] += -0.011189851;
            } else {
              result[0] += -0.023633793;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.009847272;
            } else {
              result[0] += 0.0011721178;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
            result[0] += -0.033823837;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += -0.028593678;
            } else {
              result[0] += -0.023168823;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
          result[0] += 0.048958402;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 226))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 226))) {
              result[0] += -0.0030530605;
            } else {
              result[0] += 0.009326304;
            }
          } else {
            result[0] += 0.015771423;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
            result[0] += -0.019048717;
          } else {
            result[0] += -0.00045981476;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
            result[0] += 0.036297392;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
              result[0] += 0.004649827;
            } else {
              result[0] += 0.024824895;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
            result[0] += 0.0042260056;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
              result[0] += 0.035964742;
            } else {
              result[0] += 0.05283689;
            }
          }
        } else {
          result[0] += 0.072465;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 228))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += -0.012068967;
            } else {
              result[0] += -0.027094675;
            }
          } else {
            result[0] += -0.066233106;
          }
        } else {
          result[0] += 0.025606213;
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 28))) {
            result[0] += 0.0134207625;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
              result[0] += -0.051151592;
            } else {
              result[0] += -0.013505876;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
            result[0] += 0.033789955;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 226))) {
              result[0] += -0.0063351872;
            } else {
              result[0] += 0.01952872;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 232))) {
              result[0] += 0.017679365;
            } else {
              result[0] += 0.0012839021;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 188))) {
              result[0] += 0.01832609;
            } else {
              result[0] += 0.031185871;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
              result[0] += 0.07488849;
            } else {
              result[0] += 0.046187967;
            }
          } else {
            result[0] += 0.09955382;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
            result[0] += 0.069774024;
          } else {
            result[0] += 0.10838503;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 266))) {
            result[0] += 0.05327609;
          } else {
            result[0] += 0.077984296;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
              result[0] += 0.0047698705;
            } else {
              result[0] += 0.017100412;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
              result[0] += -0.0088116955;
            } else {
              result[0] += 0.0046756053;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 46))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 138))) {
              result[0] += -0.021546138;
            } else {
              result[0] += -0.06706627;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += 0.0009891371;
            } else {
              result[0] += -0.015693253;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 76))) {
              result[0] += -0.033967104;
            } else {
              result[0] += -0.010871164;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
              result[0] += -0.01983557;
            } else {
              result[0] += -0.033842903;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += 3.485963e-05;
            } else {
              result[0] += 0.029934336;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 208))) {
              result[0] += -0.009020819;
            } else {
              result[0] += -0.0016374525;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
              result[0] += -0.00035558367;
            } else {
              result[0] += 0.031120656;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 184))) {
              result[0] += 0.0042784135;
            } else {
              result[0] += -0.003890027;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            result[0] += 0.03780892;
          } else {
            result[0] += 0.020871228;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
              result[0] += 0.032693;
            } else {
              result[0] += 0.016958345;
            }
          } else {
            result[0] += 0.056264985;
          }
        } else {
          result[0] += 0.061108418;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
        result[0] += 0.016505312;
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 198))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 58))) {
              result[0] += -0.012738267;
            } else {
              result[0] += -0.020164208;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
              result[0] += 0.0137371225;
            } else {
              result[0] += -0.010237254;
            }
          }
        } else {
          result[0] += -0.037308313;
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 112))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
              result[0] += 0.0016659134;
            } else {
              result[0] += 0.022526134;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 16))) {
              result[0] += 0.02430934;
            } else {
              result[0] += 0.010316737;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += 0.03054412;
            } else {
              result[0] += 0.013778204;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 218))) {
              result[0] += 0.062106986;
            } else {
              result[0] += 0.08695674;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
          result[0] += 0.097699985;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 266))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 226))) {
              result[0] += 0.048410773;
            } else {
              result[0] += 0.061695654;
            }
          } else {
            result[0] += 0.0699871;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 240))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
              result[0] += 0.017755916;
            } else {
              result[0] += -0.0076434105;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 80))) {
              result[0] += -0.013907449;
            } else {
              result[0] += -0.028985953;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.03517434;
            } else {
              result[0] += 0.02803059;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += -0.005040298;
            } else {
              result[0] += 0.029063394;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 18))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 252))) {
              result[0] += -0.0004860425;
            } else {
              result[0] += -0.00823578;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
              result[0] += 0.020955933;
            } else {
              result[0] += -0.005362306;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
              result[0] += -0.022735301;
            } else {
              result[0] += -0.006933318;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += 0.008230773;
            } else {
              result[0] += -0.011880973;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 232))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 34))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 192))) {
              result[0] += 0.00536417;
            } else {
              result[0] += -0.020807503;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
              result[0] += 0.029748295;
            } else {
              result[0] += 0.041834693;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 196))) {
              result[0] += -0.00087343634;
            } else {
              result[0] += -0.026689067;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 182))) {
              result[0] += -0.0014600045;
            } else {
              result[0] += 0.013319497;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 256))) {
          result[0] += 0.0012430011;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
            result[0] += -0.028792778;
          } else {
            result[0] += -0.024853468;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 4))) {
              result[0] += -0.009469046;
            } else {
              result[0] += 0.0108768735;
            }
          } else {
            result[0] += -0.025166774;
          }
        } else {
          result[0] += -0.061156314;
        }
      } else {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
              result[0] += -0.011232008;
            } else {
              result[0] += -0.025154117;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
              result[0] += 0.008252529;
            } else {
              result[0] += 0.046288077;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 6))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 220))) {
              result[0] += 0.040085025;
            } else {
              result[0] += -0.008784636;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
              result[0] += -0.019954626;
            } else {
              result[0] += 0.0048681814;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 178))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += 0.007502618;
            } else {
              result[0] += -0.0053296923;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 214))) {
              result[0] += 0.013098051;
            } else {
              result[0] += 0.022515437;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 222))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 234))) {
              result[0] += 0.02392622;
            } else {
              result[0] += 0.006717711;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 92))) {
              result[0] += 0.036931966;
            } else {
              result[0] += 0.024680862;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 192))) {
              result[0] += 0.049618512;
            } else {
              result[0] += 0.030690536;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 248))) {
              result[0] += 0.087034844;
            } else {
              result[0] += 0.06434413;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += 0.076431125;
            } else {
              result[0] += 0.050094455;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.08487356;
            } else {
              result[0] += 0.016335858;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 228))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
              result[0] += 0.0068731843;
            } else {
              result[0] += -0.027962524;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 30))) {
              result[0] += -0.005616961;
            } else {
              result[0] += -0.020151896;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 160))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
              result[0] += -0.011980721;
            } else {
              result[0] += -0.022607153;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
              result[0] += 0.045720205;
            } else {
              result[0] += 0.011621847;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 58))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
              result[0] += 0.02517539;
            } else {
              result[0] += -0.0003533575;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 76))) {
              result[0] += 0.00938051;
            } else {
              result[0] += -0.011884357;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += -0.0057867505;
            } else {
              result[0] += -0.025664538;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 208))) {
              result[0] += -0.008385401;
            } else {
              result[0] += -0.0017240072;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 266))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
              result[0] += -0.000107964406;
            } else {
              result[0] += 0.0060515343;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 244))) {
              result[0] += -0.005519016;
            } else {
              result[0] += 0.0014766084;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += 0.02811285;
          } else {
            result[0] += 0.013522627;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 152))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
              result[0] += -0;
            } else {
              result[0] += 0.015318493;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
              result[0] += 0.02693863;
            } else {
              result[0] += 0.042594276;
            }
          }
        } else {
          result[0] += 0.05915534;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
            result[0] += -0.015769325;
          } else {
            result[0] += 0.009167842;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
            result[0] += -0.066482425;
          } else {
            result[0] += -0.029134443;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 58))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
            result[0] += -0.010345658;
          } else {
            result[0] += -0.0004427325;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
            result[0] += 0.00778659;
          } else {
            result[0] += -0.052467596;
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += 0.025486384;
            } else {
              result[0] += 0.011895447;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.022794057;
            } else {
              result[0] += 0.048023786;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += 0.024896143;
            } else {
              result[0] += 0.010721759;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += 0.04730309;
            } else {
              result[0] += 0.07061689;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
          result[0] += 0.079474784;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 6))) {
              result[0] += 0.050025214;
            } else {
              result[0] += 0.03944496;
            }
          } else {
            result[0] += 0.05712953;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 54))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0070698797;
            } else {
              result[0] += -0.009966139;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += 0.015061869;
            } else {
              result[0] += -0.0023575744;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
              result[0] += -0.021192541;
            } else {
              result[0] += -0.00409846;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 156))) {
              result[0] += -0.01823308;
            } else {
              result[0] += 0.00014080759;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 206))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 100))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
              result[0] += 0.02206879;
            } else {
              result[0] += -0.00015570642;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
              result[0] += -0.015682366;
            } else {
              result[0] += -0.006868744;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
              result[0] += 0.0135331275;
            } else {
              result[0] += -0.023557609;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
              result[0] += 0.04654594;
            } else {
              result[0] += 0.0058658714;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
            result[0] += -0.016888324;
          } else {
            result[0] += -0.00063805765;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
              result[0] += 0.029826907;
            } else {
              result[0] += 0.007284993;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
              result[0] += 0.001150948;
            } else {
              result[0] += 0.0069423555;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 152))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += -0.0042433054;
            } else {
              result[0] += 0.011705535;
            }
          } else {
            result[0] += 0.025982574;
          }
        } else {
          result[0] += 0.048386928;
        }
      }
    }
  }
  if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 206))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += -0.042859327;
            } else {
              result[0] += -0.0044231373;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
              result[0] += 0.049170107;
            } else {
              result[0] += 0.035520416;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
              result[0] += -0.0009743745;
            } else {
              result[0] += 0.009931285;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
              result[0] += -0.0017889539;
            } else {
              result[0] += -0.007930504;
            }
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 226))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 204))) {
              result[0] += 0.0063184327;
            } else {
              result[0] += -0.015213072;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += 0.014672543;
            } else {
              result[0] += 0.026392935;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 154))) {
              result[0] += -0.037449803;
            } else {
              result[0] += -0.053623743;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
              result[0] += 0.002925502;
            } else {
              result[0] += -0.010573565;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 246))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 264))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 152))) {
              result[0] += 0.00568271;
            } else {
              result[0] += -0.004481957;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 212))) {
              result[0] += 0.053150166;
            } else {
              result[0] += 0.014763187;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 194))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 80))) {
              result[0] += -0.008710187;
            } else {
              result[0] += 0.075820915;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
              result[0] += -0.023372917;
            } else {
              result[0] += -0.012099934;
            }
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 204))) {
            result[0] += 0.037964396;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 206))) {
              result[0] += 0.0036865615;
            } else {
              result[0] += 0.021090202;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
            result[0] += -0.025623156;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 200))) {
              result[0] += -0.004627872;
            } else {
              result[0] += 0.004686881;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 262))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 248))) {
          result[0] += 0.0714253;
        } else {
          result[0] += 0.048554998;
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 232))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 110))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 230))) {
              result[0] += 0.02269533;
            } else {
              result[0] += 0.014301536;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 220))) {
              result[0] += 0.0025220858;
            } else {
              result[0] += 0.011460328;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 56))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += 0.019217266;
            } else {
              result[0] += 0.03249597;
            }
          } else {
            result[0] += 0.035739582;
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 248))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
              result[0] += -0.023269765;
            } else {
              result[0] += -0.01868113;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 180))) {
              result[0] += -0.0065037594;
            } else {
              result[0] += -0.022253536;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 272))) {
              result[0] += 0.0036995602;
            } else {
              result[0] += 0.0520568;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 146))) {
              result[0] += 0.02432552;
            } else {
              result[0] += -0.01594592;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
              result[0] += 0.0497043;
            } else {
              result[0] += 0.022733098;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.055147793;
            } else {
              result[0] += 0.041854013;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 166))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
              result[0] += 0.0019326921;
            } else {
              result[0] += -0.013451728;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 280))) {
              result[0] += 0.029732933;
            } else {
              result[0] += 0.009648888;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 206))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
            result[0] += -0.002109048;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 92))) {
              result[0] += 0.04531033;
            } else {
              result[0] += 0.028058738;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
              result[0] += -0.0014130161;
            } else {
              result[0] += 0.009621711;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
              result[0] += -0.0014670437;
            } else {
              result[0] += -0.007695043;
            }
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 226))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 156))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 192))) {
              result[0] += 0.007604044;
            } else {
              result[0] += -0.0070991814;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 220))) {
              result[0] += 0.020077487;
            } else {
              result[0] += 0.006828181;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 154))) {
              result[0] += -0.03365531;
            } else {
              result[0] += -0.049779564;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
              result[0] += 0.0017921542;
            } else {
              result[0] += -0.009079668;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 246))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 260))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 152))) {
              result[0] += 0.005504231;
            } else {
              result[0] += -0.0039019831;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 212))) {
              result[0] += 0.044337142;
            } else {
              result[0] += 0.0070573166;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 194))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
              result[0] += 0.028603448;
            } else {
              result[0] += -0.0078216465;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
              result[0] += 0.015729161;
            } else {
              result[0] += -0.016172249;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 204))) {
            result[0] += 0.03005304;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
              result[0] += 0.016138185;
            } else {
              result[0] += 0.033333454;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
            result[0] += -0.02334422;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
              result[0] += 0.0137236165;
            } else {
              result[0] += 0.003765919;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 260))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.06424896;
        } else {
          result[0] += 0.045115866;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 232))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 110))) {
            result[0] += 0.020254442;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
              result[0] += 0.003927156;
            } else {
              result[0] += -0.00030379518;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 56))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += 0.01626153;
            } else {
              result[0] += 0.030240763;
            }
          } else {
            result[0] += 0.031179471;
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
          result[0] += -0.019889602;
        } else {
          result[0] += -0.006115454;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
              result[0] += 0.02490984;
            } else {
              result[0] += 0.0096712075;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 228))) {
              result[0] += -0.016214207;
            } else {
              result[0] += 0.0054753;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 162))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.04828751;
            } else {
              result[0] += 0.02183148;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
              result[0] += 0.095508024;
            } else {
              result[0] += 0.037357938;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 52))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
            result[0] += 0.024429439;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += -0.032644305;
            } else {
              result[0] += -0.007228152;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 92))) {
            result[0] += -0;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
              result[0] += -0.0005319937;
            } else {
              result[0] += 0.04273183;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += -0;
            } else {
              result[0] += -0.015805844;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 220))) {
              result[0] += 0.028671984;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 0))) {
            result[0] += -0.015004277;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 44))) {
              result[0] += -0.043184254;
            } else {
              result[0] += -0.015434514;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
              result[0] += 0.027698422;
            } else {
              result[0] += 0.019098751;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 110))) {
              result[0] += 0.0024283626;
            } else {
              result[0] += 0.019441115;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 58))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.07004071;
            } else {
              result[0] += 0.022275856;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 80))) {
              result[0] += -0.0013316347;
            } else {
              result[0] += 0.012960051;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 224))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += 0.06203411;
            } else {
              result[0] += 0.035423633;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.057469275;
            } else {
              result[0] += 0.009981114;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 212))) {
              result[0] += 0.023559755;
            } else {
              result[0] += 0.035965245;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
              result[0] += 0.058553632;
            } else {
              result[0] += 0.04478031;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
              result[0] += 0.01235016;
            } else {
              result[0] += -0.0039201183;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 80))) {
              result[0] += -0.007764698;
            } else {
              result[0] += -0.019835753;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.006487711;
            } else {
              result[0] += 0.018991722;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += -0.0039943634;
            } else {
              result[0] += 0.024546368;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 154))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 170))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 90))) {
              result[0] += -0.0147830695;
            } else {
              result[0] += -0.008563632;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
              result[0] += -0.030682612;
            } else {
              result[0] += 0.017968642;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 166))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
              result[0] += 0.068739615;
            } else {
              result[0] += 0.00021970407;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
              result[0] += -0.0028976346;
            } else {
              result[0] += -0.009871369;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 236))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
          result[0] += -0.021564588;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
              result[0] += 0.0040140515;
            } else {
              result[0] += 0.016122099;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 210))) {
              result[0] += -0.0033681232;
            } else {
              result[0] += 0.0073165386;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 240))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 226))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
              result[0] += -0.025544493;
            } else {
              result[0] += -0.01873306;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 230))) {
              result[0] += -0.019029168;
            } else {
              result[0] += 0.004258202;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 254))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 212))) {
              result[0] += -0.00011749373;
            } else {
              result[0] += -0.010187042;
            }
          } else {
            result[0] += 0.037634984;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
          result[0] += -0.0064992644;
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 146))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 144))) {
              result[0] += 0.022600017;
            } else {
              result[0] += -0.0016712642;
            }
          } else {
            result[0] += 0.025729222;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 20))) {
            result[0] += -0.007915151;
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
              result[0] += 0.0005776281;
            } else {
              result[0] += 0.031780552;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
            result[0] += -0.017479556;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 178))) {
              result[0] += -0.04499294;
            } else {
              result[0] += -0.01988611;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += 0.008722408;
            } else {
              result[0] += 0.032495603;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += -0.0015038172;
            } else {
              result[0] += 0.009706586;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += 0.08136292;
            } else {
              result[0] += 0.015997507;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
              result[0] += 0.03901923;
            } else {
              result[0] += 0.0564677;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 226))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            result[0] += 0.0416153;
          } else {
            result[0] += 0.026928617;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
            result[0] += 0.041233912;
          } else {
            result[0] += 0.05209862;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 240))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
              result[0] += 0.00972145;
            } else {
              result[0] += 0.04575323;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
              result[0] += 0.02476694;
            } else {
              result[0] += -0.03629762;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.021036565;
            } else {
              result[0] += -0.045760524;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 224))) {
              result[0] += 0.007162188;
            } else {
              result[0] += -0.018060943;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 22))) {
              result[0] += -0.007815191;
            } else {
              result[0] += -0.017675815;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
              result[0] += -0.0029981257;
            } else {
              result[0] += 0.020084158;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
              result[0] += 0.02050746;
            } else {
              result[0] += -0.0026271623;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
              result[0] += -0.011671978;
            } else {
              result[0] += -0.0055001257;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 228))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
            result[0] += -0.019566214;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
              result[0] += 0.0054924428;
            } else {
              result[0] += -0.00093082106;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
            result[0] += 0.0126124;
          } else {
            result[0] += 0.032813326;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 240))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 272))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
              result[0] += -0.023916144;
            } else {
              result[0] += -0.017258545;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.05414641;
            } else {
              result[0] += 0.000113278526;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 254))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
              result[0] += -0.00055813155;
            } else {
              result[0] += -0.008351603;
            }
          } else {
            result[0] += 0.032734703;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
              result[0] += -0.0139853405;
            } else {
              result[0] += 0.003684853;
            }
          } else {
            result[0] += 0.008061817;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
            result[0] += -0.06050358;
          } else {
            result[0] += -0.0023646136;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
          result[0] += 0.028764948;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += -0.028082406;
            } else {
              result[0] += -0.0058664638;
            }
          } else {
            result[0] += 0.023923343;
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 170))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
              result[0] += 0.009201097;
            } else {
              result[0] += -0.010257622;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.012351765;
            } else {
              result[0] += 0.08077949;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
            result[0] += 0.037493166;
          } else {
            result[0] += 0.052167382;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
            result[0] += 0.036415916;
          } else {
            result[0] += 0.046912543;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 266))) {
            result[0] += 0.023727143;
          } else {
            result[0] += 0.036457557;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 208))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
              result[0] += -0.002685583;
            } else {
              result[0] += -0.006960124;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 90))) {
              result[0] += 0.04494014;
            } else {
              result[0] += 0.0043056607;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 220))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
              result[0] += -0.0039729676;
            } else {
              result[0] += -0.014122757;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 178))) {
              result[0] += -0.025312701;
            } else {
              result[0] += -0.014867464;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          result[0] += 0.015924158;
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 190))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 184))) {
              result[0] += -0.0075506503;
            } else {
              result[0] += 0.010475333;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
              result[0] += -0.027276382;
            } else {
              result[0] += -0.004030099;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 98))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
              result[0] += 0.014477782;
            } else {
              result[0] += -0.015422763;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.03197563;
            } else {
              result[0] += 0.020037709;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 134))) {
            result[0] += -0.019650823;
          } else {
            result[0] += 0.002236197;
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 150))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
              result[0] += -0.02055483;
            } else {
              result[0] += -0.0014581191;
            }
          } else {
            result[0] += -0.016465666;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 242))) {
              result[0] += -0.05338929;
            } else {
              result[0] += -0.014811342;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
              result[0] += 0.01168551;
            } else {
              result[0] += 0.0010410798;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += -0.005663588;
            } else {
              result[0] += 0.012237726;
            }
          } else {
            result[0] += -0.015190408;
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
            result[0] += -0.048400734;
          } else {
            result[0] += -0.01567449;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
          result[0] += 0.034811612;
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 166))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
              result[0] += -0.005241947;
            } else {
              result[0] += 0.0036905694;
            }
          } else {
            result[0] += 0.027789101;
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 172))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 168))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
              result[0] += -0.0023323249;
            } else {
              result[0] += 0.008576729;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += -0.02920859;
            } else {
              result[0] += -0.005879428;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 236))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
              result[0] += 0.01825313;
            } else {
              result[0] += 0.01146;
            }
          } else {
            result[0] += -0.0015520846;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
            result[0] += 0.050887883;
          } else {
            result[0] += 0.0008823246;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
            result[0] += 0.04027082;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 252))) {
              result[0] += 0.02058533;
            } else {
              result[0] += 0.039607793;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 28))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0065161586;
            } else {
              result[0] += -0.007997109;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
              result[0] += 0.005459733;
            } else {
              result[0] += -0.0032214918;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
              result[0] += -0.015889943;
            } else {
              result[0] += -0.054882165;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 160))) {
              result[0] += 0.013258442;
            } else {
              result[0] += -0.008985414;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
              result[0] += 0.03941143;
            } else {
              result[0] += 0.01124495;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 100))) {
              result[0] += -0.01534603;
            } else {
              result[0] += 0.0053413953;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 80))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 236))) {
              result[0] += -0.017410088;
            } else {
              result[0] += 0.013132173;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
              result[0] += -0.0060942764;
            } else {
              result[0] += -0.0017843047;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 222))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
              result[0] += 0.035340164;
            } else {
              result[0] += 0.0032424498;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += 0.02138096;
            } else {
              result[0] += 0.011583105;
            }
          }
        } else {
          result[0] += -0.0026775673;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 152))) {
              result[0] += 0.0032441877;
            } else {
              result[0] += 0.013359073;
            }
          } else {
            result[0] += 0.030153606;
          }
        } else {
          result[0] += 0.036456186;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
            result[0] += -0.0117252795;
          } else {
            result[0] += 0.0043645753;
          }
        } else {
          result[0] += -0.041562933;
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          result[0] += 0.0008401942;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            result[0] += -0.01901227;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
              result[0] += 0.014339037;
            } else {
              result[0] += -0.00501071;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 16))) {
              result[0] += 0.012573126;
            } else {
              result[0] += 0.0028887743;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 38))) {
              result[0] += 0.046177443;
            } else {
              result[0] += 0.015051349;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += 0.010199251;
            } else {
              result[0] += -0.039779864;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 32))) {
              result[0] += -0.004799234;
            } else {
              result[0] += 0.008129753;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            result[0] += 0.04275175;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
              result[0] += -0.008011124;
            } else {
              result[0] += 0.002239405;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 226))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 268))) {
              result[0] += 0.017378718;
            } else {
              result[0] += 0.03415826;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 218))) {
              result[0] += 0.02640838;
            } else {
              result[0] += 0.039306883;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0055097975;
            } else {
              result[0] += -0.007307568;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += 0.0022932815;
            } else {
              result[0] += -0.0064788023;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
              result[0] += -0.014033759;
            } else {
              result[0] += -0.049122956;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 90))) {
              result[0] += 0.005159875;
            } else {
              result[0] += -0.009883177;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
              result[0] += -0.013826482;
            } else {
              result[0] += 0.022371221;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
              result[0] += -0.0141291395;
            } else {
              result[0] += 0.004760613;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 132))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
              result[0] += -0.007064409;
            } else {
              result[0] += 0.023824794;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
              result[0] += -0.00016228954;
            } else {
              result[0] += -0.007653934;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 222))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 208))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
              result[0] += 0.004959669;
            } else {
              result[0] += -0.001015551;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 212))) {
              result[0] += 0.027703479;
            } else {
              result[0] += 0.0036117423;
            }
          }
        } else {
          result[0] += -0.0026553113;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
              result[0] += 0.013676906;
            } else {
              result[0] += -0.0007326856;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
              result[0] += -0;
            } else {
              result[0] += 0.030973986;
            }
          }
        } else {
          result[0] += 0.029461041;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 8))) {
            result[0] += -0.009251536;
          } else {
            result[0] += 0.009845628;
          }
        } else {
          result[0] += -0.04083568;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.029976163;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            result[0] += -0.027244702;
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 166))) {
              result[0] += -0.004565258;
            } else {
              result[0] += 0.025788173;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 254))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 34))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
              result[0] += 0.011977387;
            } else {
              result[0] += 0.00021163402;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
              result[0] += 0.038581323;
            } else {
              result[0] += 0.013672948;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += 0.0043165404;
            } else {
              result[0] += -0.035845544;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
              result[0] += -0.0024726556;
            } else {
              result[0] += 0.008141091;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            result[0] += 0.03638802;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 166))) {
              result[0] += 0.0001506592;
            } else {
              result[0] += 0.009602265;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 218))) {
              result[0] += 0.022467962;
            } else {
              result[0] += 0.03540038;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += 0.06481004;
            } else {
              result[0] += 0.017584896;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 208))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 140))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 104))) {
              result[0] += 0.045290753;
            } else {
              result[0] += -0.01791346;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += -0.058552768;
            } else {
              result[0] += 0.0349788;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 256))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 198))) {
              result[0] += 0.0336964;
            } else {
              result[0] += 0.0018466607;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 176))) {
              result[0] += -0.0049308436;
            } else {
              result[0] += -0.012199649;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 86))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 100))) {
              result[0] += -0.007085;
            } else {
              result[0] += -0.016483508;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += 0.006847045;
            } else {
              result[0] += -0.0070480793;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += -0.0041096415;
            } else {
              result[0] += 0.0021135767;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
              result[0] += -0.0060591055;
            } else {
              result[0] += -0.0011625281;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 138))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 28))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
              result[0] += 0.02211509;
            } else {
              result[0] += 0.060023166;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
              result[0] += -0.013461961;
            } else {
              result[0] += 0.04670935;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
            result[0] += -0;
          } else {
            result[0] += 0.025248567;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 154))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 148))) {
              result[0] += -0.008415373;
            } else {
              result[0] += 0.013447831;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 138))) {
              result[0] += -0.016349984;
            } else {
              result[0] += -0.030504474;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += 0.00021573504;
            } else {
              result[0] += -0.013515745;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += 8.818205e-05;
            } else {
              result[0] += 0.010008925;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 18))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
          result[0] += 0.0037289502;
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 124))) {
            result[0] += 0.054951753;
          } else {
            result[0] += -0.0039979555;
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
            result[0] += -0.00068969437;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
              result[0] += -0.012617968;
            } else {
              result[0] += -0.027266933;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 160))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 48))) {
              result[0] += 0.0068334453;
            } else {
              result[0] += -0.0039404687;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 30))) {
              result[0] += -0;
            } else {
              result[0] += 0.039604023;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 178))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 36))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 154))) {
              result[0] += -0.0074695237;
            } else {
              result[0] += 0.0059394534;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 218))) {
              result[0] += 0.02302828;
            } else {
              result[0] += -0.047223628;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 156))) {
              result[0] += -0.016071025;
            } else {
              result[0] += 0.0044617057;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
              result[0] += -0.0016089712;
            } else {
              result[0] += 0.0069275023;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.030206159;
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 228))) {
              result[0] += 0.010797313;
            } else {
              result[0] += -0.002882598;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
              result[0] += 0.020259693;
            } else {
              result[0] += 0.0355315;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 192))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 78))) {
              result[0] += -0.0044226176;
            } else {
              result[0] += -0.02530554;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
              result[0] += 0.008079353;
            } else {
              result[0] += -0.003107704;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 208))) {
              result[0] += -0.026262758;
            } else {
              result[0] += -0.013569482;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
              result[0] += -0.007361694;
            } else {
              result[0] += 0.01589146;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 218))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 186))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
              result[0] += 0.0066897767;
            } else {
              result[0] += -0.03870764;
            }
          } else {
            result[0] += 0.022245135;
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 156))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 46))) {
              result[0] += 0.0011395493;
            } else {
              result[0] += -0.0057673557;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
              result[0] += 0.022744471;
            } else {
              result[0] += 0.008478357;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 222))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += 0.028410172;
            } else {
              result[0] += 0.014004901;
            }
          } else {
            result[0] += 0.0010523595;
          }
        } else {
          result[0] += 0.02651517;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 176))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 226))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 228))) {
              result[0] += -0.0015872531;
            } else {
              result[0] += -0.017748317;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += -0.0032637853;
            } else {
              result[0] += 0.010829237;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 242))) {
            result[0] += 0.034405578;
          } else {
            result[0] += 0.00891859;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 20))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 254))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 214))) {
            result[0] += 0.007226427;
          } else {
            result[0] += -0.012499331;
          }
        } else {
          result[0] += 0.04246553;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 88))) {
              result[0] += 0.002265221;
            } else {
              result[0] += -0.0032362868;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.011394487;
            } else {
              result[0] += -0.0008440514;
            }
          }
        } else {
          result[0] += -0.028067807;
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
              result[0] += 0.08105516;
            } else {
              result[0] += 0.018269602;
            }
          } else {
            result[0] += 0.015147194;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
              result[0] += 0.014643987;
            } else {
              result[0] += 0.00034420882;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
              result[0] += 0.019848578;
            } else {
              result[0] += 0.0055640773;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.0281445;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
            result[0] += 0.016479755;
          } else {
            result[0] += 0.023855714;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 252))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
              result[0] += 0.0043632514;
            } else {
              result[0] += -0.0032153563;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += -0.0008953023;
            } else {
              result[0] += -0.012973676;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 224))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
              result[0] += 0.06142653;
            } else {
              result[0] += 0.014740917;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
              result[0] += 0.036935415;
            } else {
              result[0] += -0.0027422463;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 206))) {
              result[0] += -0.043081205;
            } else {
              result[0] += -0.020138515;
            }
          } else {
            result[0] += -0.012991915;
          }
        } else {
          result[0] += 0.005466835;
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
              result[0] += -0.00065770803;
            } else {
              result[0] += 0.020950794;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 98))) {
              result[0] += -0.0020690185;
            } else {
              result[0] += 0.0022426536;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += 0.01779854;
          } else {
            result[0] += 0.007740788;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
            result[0] += 0.010289395;
          } else {
            result[0] += 0.022736436;
          }
        } else {
          result[0] += 0.02622193;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 52))) {
            result[0] += -0.0030147363;
          } else {
            result[0] += -0.01984936;
          }
        } else {
          result[0] += -0.014292531;
        }
      } else {
        result[0] += 0.019834789;
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 12))) {
            result[0] += 0.012920295;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
              result[0] += 0.07497602;
            } else {
              result[0] += 0.01954972;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 16))) {
              result[0] += 0.010222583;
            } else {
              result[0] += 0.00049927505;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
              result[0] += -0.0032804606;
            } else {
              result[0] += 0.0070060194;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.024341205;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
            result[0] += 0.015133323;
          } else {
            result[0] += 0.021515321;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
              result[0] += 0.0021268248;
            } else {
              result[0] += -0.004917339;
            }
          } else {
            result[0] += 0.020086113;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
            result[0] += -0.023110032;
          } else {
            result[0] += -0.008982231;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 206))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
              result[0] += -0.01757216;
            } else {
              result[0] += -5.0931576e-05;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
              result[0] += 0.017951632;
            } else {
              result[0] += 0.0057268315;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 62))) {
              result[0] += 0.0030992;
            } else {
              result[0] += -0.010930783;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += -0.00038469338;
            } else {
              result[0] += -0.004318268;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
              result[0] += -0;
            } else {
              result[0] += 0.017621709;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
              result[0] += 0.0036806122;
            } else {
              result[0] += -4.4085446e-05;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += 0.021112496;
          } else {
            result[0] += 0.007883241;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
            result[0] += 0.007880076;
          } else {
            result[0] += 0.02039106;
          }
        } else {
          result[0] += 0.023736782;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 14))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
          result[0] += -0;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
            result[0] += 0.022109;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
              result[0] += -0.011896599;
            } else {
              result[0] += -0.02695978;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 28))) {
          result[0] += 0.017046606;
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 86))) {
            result[0] += -0.022864131;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 170))) {
              result[0] += 0.01655387;
            } else {
              result[0] += -0.0029186176;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
              result[0] += 0.006224805;
            } else {
              result[0] += -0.0006160491;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
              result[0] += 0.03912334;
            } else {
              result[0] += 0.0064465962;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += 0.03801068;
            } else {
              result[0] += -0;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
              result[0] += -0.0056696637;
            } else {
              result[0] += 0.002394879;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
              result[0] += 0.0026905416;
            } else {
              result[0] += 0.014434603;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 18))) {
              result[0] += -0.0066401684;
            } else {
              result[0] += -0.02399191;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 216))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
              result[0] += 0.01420292;
            } else {
              result[0] += 0.007666865;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 258))) {
              result[0] += 0.015625248;
            } else {
              result[0] += 0.035386283;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 240))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 168))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 164))) {
              result[0] += -0.0026949244;
            } else {
              result[0] += 0.010536998;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 172))) {
              result[0] += -0.02333055;
            } else {
              result[0] += -0.004258865;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 190))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 256))) {
              result[0] += 0.010322237;
            } else {
              result[0] += -0.004513755;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 194))) {
              result[0] += -0.013793135;
            } else {
              result[0] += 0.00041879248;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
            result[0] += -0.0012282294;
          } else {
            result[0] += 0.012586914;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
            result[0] += -0.013370514;
          } else {
            result[0] += -0.009561422;
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
            result[0] += 0.020673957;
          } else {
            result[0] += 0.01021281;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            result[0] += 0.029428396;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
              result[0] += -0.0008754981;
            } else {
              result[0] += -0.0100570135;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 198))) {
              result[0] += 0.016708953;
            } else {
              result[0] += 0.009689736;
            }
          } else {
            result[0] += 0.00040766338;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 252))) {
            result[0] += 0.024215741;
          } else {
            result[0] += 0.012051038;
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
              result[0] += 0.0013062642;
            } else {
              result[0] += 0.019394014;
            }
          } else {
            result[0] += -0.0104289735;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 48))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.05682643;
            } else {
              result[0] += 0.026358023;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 126))) {
              result[0] += 0.0067068855;
            } else {
              result[0] += -0.0033787123;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 12))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 96))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += -0.017422771;
            } else {
              result[0] += -0.013566106;
            }
          } else {
            result[0] += 0.05098955;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 264))) {
              result[0] += 0.0021389506;
            } else {
              result[0] += 0.022104675;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 196))) {
              result[0] += -0.0034482577;
            } else {
              result[0] += -0.04897159;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 94))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
              result[0] += 0.04547104;
            } else {
              result[0] += -0.0036854;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.031185264;
            } else {
              result[0] += -0.00029240636;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
            result[0] += 0.048903078;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
              result[0] += -0.012713486;
            } else {
              result[0] += -0.0045942473;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 32))) {
              result[0] += -0.024384957;
            } else {
              result[0] += 0.010758282;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.030021293;
            } else {
              result[0] += 0.017209774;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 220))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 88))) {
              result[0] += -0.0025992983;
            } else {
              result[0] += 0.0014243874;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.014005645;
            } else {
              result[0] += 0.00015922675;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 184))) {
            result[0] += -0.0023512202;
          } else {
            result[0] += 0.00030087968;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
              result[0] += 0.015333219;
            } else {
              result[0] += 0.050305735;
            }
          } else {
            result[0] += 0.0038954068;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 220))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
            result[0] += 0.005740883;
          } else {
            result[0] += -0.011134521;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
              result[0] += -0.0102155125;
            } else {
              result[0] += -0.01861443;
            }
          } else {
            result[0] += -0.008878123;
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 36))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 58))) {
            result[0] += 0.07084565;
          } else {
            result[0] += 0.019467566;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 106))) {
              result[0] += -0.005714934;
            } else {
              result[0] += 0.0072955615;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 212))) {
              result[0] += 0.0217909;
            } else {
              result[0] += 0.008551905;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
          result[0] += 0.016447848;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
              result[0] += 0.028782597;
            } else {
              result[0] += -0.027478626;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
              result[0] += 0.007580962;
            } else {
              result[0] += 0.0014997128;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
      if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 214))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 198))) {
            result[0] += -0.0022094585;
          } else {
            result[0] += -0.02451445;
          }
        } else {
          result[0] += 0.024397848;
        }
      } else {
        result[0] += -0.01616343;
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
              result[0] += 0.011657309;
            } else {
              result[0] += 0.0012069183;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.040601656;
            } else {
              result[0] += 0.009451126;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
            result[0] += -0.022806326;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
              result[0] += 0.03292076;
            } else {
              result[0] += 0.001937497;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 180))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += 0.0035285496;
            } else {
              result[0] += -0.0058691064;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
              result[0] += 0.012688639;
            } else {
              result[0] += 0.005491004;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 146))) {
            result[0] += 0.0044394704;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 192))) {
              result[0] += 0.012408857;
            } else {
              result[0] += 0.019069476;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 0))) {
              result[0] += 0.006046584;
            } else {
              result[0] += -0.015792418;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.027976854;
            } else {
              result[0] += 0.012421743;
            }
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            result[0] += 0.054596562;
          } else {
            result[0] += 0.01953054;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
              result[0] += -0.003245589;
            } else {
              result[0] += 0.013587843;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.021004315;
            } else {
              result[0] += -0.008656485;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
              result[0] += 0.01555752;
            } else {
              result[0] += -0.0006588964;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
              result[0] += -0.0020559;
            } else {
              result[0] += -0.010479237;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
          result[0] += 0.021051032;
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 220))) {
            result[0] += 0.006872528;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
              result[0] += -0.00075457775;
            } else {
              result[0] += -0.008546843;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
          result[0] += 0.009004579;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 2))) {
            result[0] += 0.013524537;
          } else {
            result[0] += 0.02246733;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 202))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 28))) {
            result[0] += -0.004806658;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += -0.02193972;
            } else {
              result[0] += -0.0058797724;
            }
          }
        } else {
          result[0] += -0.023569103;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 264))) {
          result[0] += 0.016511796;
        } else {
          result[0] += -0.0023302487;
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.004636476;
            } else {
              result[0] += 0.03744141;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 154))) {
              result[0] += -0.0021920146;
            } else {
              result[0] += 0.0044151954;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
            result[0] += 0.013442422;
          } else {
            result[0] += 0.025949672;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 252))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
            result[0] += 0.012082203;
          } else {
            result[0] += 0.017159306;
          }
        } else {
          result[0] += 0.018528607;
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 28))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
              result[0] += 0.012809316;
            } else {
              result[0] += 0.0020744025;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
              result[0] += 0.02068119;
            } else {
              result[0] += 0.012666582;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.025207102;
            } else {
              result[0] += -0.0042135157;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
              result[0] += 0.019998292;
            } else {
              result[0] += -0.011731009;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          result[0] += 0.04935061;
        } else {
          result[0] += 0.00985283;
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 2))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 10))) {
              result[0] += -0.0114418;
            } else {
              result[0] += 0.001085467;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 228))) {
              result[0] += 0.005336402;
            } else {
              result[0] += 0.040240493;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
            result[0] += -0.0003519562;
          } else {
            result[0] += -0.016158544;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 22))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 186))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += -0.052682735;
            } else {
              result[0] += -0.024834817;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.0076021785;
            } else {
              result[0] += -0.009852809;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 24))) {
            result[0] += 0.07371118;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
              result[0] += -0.0062063127;
            } else {
              result[0] += -0.0016409116;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 166))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 138))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 96))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += -0.0056977295;
            } else {
              result[0] += -0.0020788263;
            }
          } else {
            result[0] += 0.013674322;
          }
        } else {
          result[0] += -0.012557744;
        }
      } else {
        result[0] += 0.020858606;
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 100))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
              result[0] += 0.004243218;
            } else {
              result[0] += -0.0012915577;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 110))) {
              result[0] += 0.012041297;
            } else {
              result[0] += 0.0036917683;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 226))) {
              result[0] += 0.021245888;
            } else {
              result[0] += 0.010770134;
            }
          } else {
            result[0] += 0.026461909;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 252))) {
          result[0] += 0.011941756;
        } else {
          result[0] += 0.01697006;
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 190))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
              result[0] += 0.0059637506;
            } else {
              result[0] += 0.0009332538;
            }
          } else {
            result[0] += -0.01690738;
          }
        } else {
          result[0] += 0.018049363;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 30))) {
              result[0] += -0.003757357;
            } else {
              result[0] += -0.010965187;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 138))) {
              result[0] += 0.054250848;
            } else {
              result[0] += -0.00039252196;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 42))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += 0.01451879;
            } else {
              result[0] += 0.037112333;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 36))) {
              result[0] += -0.017370671;
            } else {
              result[0] += -0.0013852807;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 222))) {
            result[0] += 0.009491138;
          } else {
            result[0] += 0.03599734;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
              result[0] += 0.0021581096;
            } else {
              result[0] += -0.0014275697;
            }
          } else {
            result[0] += 0.009684578;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
            result[0] += 0.009782214;
          } else {
            result[0] += -0.0025362705;
          }
        } else {
          result[0] += 0.015987474;
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
        result[0] += 0.031121805;
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
            result[0] += -0.0017203381;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 10))) {
              result[0] += 0.030785752;
            } else {
              result[0] += -0.005102741;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
              result[0] += -0.005013264;
            } else {
              result[0] += 0.027989835;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
              result[0] += -0.024730321;
            } else {
              result[0] += -0.0032983057;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 244))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
          result[0] += 0.009897011;
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 94))) {
              result[0] += -0.0021574632;
            } else {
              result[0] += 0.057048213;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
              result[0] += 0.010313834;
            } else {
              result[0] += 0.002621749;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          result[0] += 0.015189426;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
            result[0] += 0.009792722;
          } else {
            result[0] += 0.014417933;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 204))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 0))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.0014002819;
            } else {
              result[0] += 0.006653227;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
              result[0] += -0.023881285;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            result[0] += 0.018696984;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.0133892;
            } else {
              result[0] += 0.008133677;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          result[0] += 0.043852713;
        } else {
          result[0] += 0.0058281594;
        }
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 212))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 144))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 174))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 214))) {
              result[0] += -0.0026800435;
            } else {
              result[0] += 0.0020921542;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += 0.0018816863;
            } else {
              result[0] += -0.027228499;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            result[0] += -0.014789981;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 150))) {
              result[0] += -0.009289259;
            } else {
              result[0] += -0.0037157678;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 158))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += -0.009539143;
            } else {
              result[0] += 0.006927042;
            }
          } else {
            result[0] += 0.039872702;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 160))) {
            result[0] += -0.039126202;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 166))) {
              result[0] += -0.002670629;
            } else {
              result[0] += 0.001439321;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
            result[0] += -5.299445e-05;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.019934397;
            } else {
              result[0] += -0.0012183964;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 154))) {
            result[0] += -0.0013647849;
          } else {
            result[0] += 0.038733196;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
              result[0] += 0.008902398;
            } else {
              result[0] += -0.00011572992;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 14))) {
              result[0] += 3.4862278e-06;
            } else {
              result[0] += 0.012706413;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 58))) {
              result[0] += 0.007615597;
            } else {
              result[0] += -0.0051007196;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 48))) {
              result[0] += 0.015438475;
            } else {
              result[0] += 0.0018905745;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 246))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 230))) {
              result[0] += 0.008593912;
            } else {
              result[0] += 0.014445068;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.00014397338;
            } else {
              result[0] += 0.0083941175;
            }
          }
        } else {
          result[0] += -0.004308312;
        }
      } else {
        result[0] += 0.046102088;
      }
    }
  } else {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 190))) {
              result[0] += -0.0013631996;
            } else {
              result[0] += -0.011290415;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 200))) {
              result[0] += 0.00060525746;
            } else {
              result[0] += 0.013914026;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 248))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += -0.023959834;
            } else {
              result[0] += -0.010296635;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
              result[0] += 0.008832153;
            } else {
              result[0] += 0.0023550882;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
            result[0] += -0.015583684;
          } else {
            result[0] += 0.010421765;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
            result[0] += -0.00011256654;
          } else {
            result[0] += -0.02525649;
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 232))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 230))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 44))) {
              result[0] += 0.005643142;
            } else {
              result[0] += -0.00010354039;
            }
          } else {
            result[0] += -0.015879504;
          }
        } else {
          result[0] += 0.014167144;
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 246))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 58))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.049543332;
            } else {
              result[0] += 0.0046836296;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
              result[0] += -0.017134476;
            } else {
              result[0] += -0.007789578;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
            result[0] += -0.008826947;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 250))) {
              result[0] += -0.0007550689;
            } else {
              result[0] += 0.0071488456;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 188))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 238))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 106))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
              result[0] += -0.019138163;
            } else {
              result[0] += 0.008177842;
            }
          } else {
            result[0] += 0.03202384;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.03914841;
            } else {
              result[0] += 0.027094522;
            }
          } else {
            result[0] += 0.0053101527;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.0403822;
            } else {
              result[0] += 0.013740751;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
              result[0] += -0.030392349;
            } else {
              result[0] += 0.0028747516;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 80))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 204))) {
              result[0] += -0.008877286;
            } else {
              result[0] += 0.012316696;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 118))) {
              result[0] += 0.004029528;
            } else {
              result[0] += -0.00029411144;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 2))) {
            result[0] += 0.011664437;
          } else {
            result[0] += 0.0006082279;
          }
        } else {
          result[0] += -0.006005393;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 166))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
              result[0] += 0.09773924;
            } else {
              result[0] += 0.02065113;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 200))) {
              result[0] += 0.0035237612;
            } else {
              result[0] += 0.009829774;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 252))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 208))) {
              result[0] += -0.007697291;
            } else {
              result[0] += 0.0015825549;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
              result[0] += 0.0036124797;
            } else {
              result[0] += 0.014532133;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
              result[0] += 0.0012666411;
            } else {
              result[0] += -0.03202522;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
              result[0] += 0.020200811;
            } else {
              result[0] += 0.004590917;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 28))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
              result[0] += 0.03727207;
            } else {
              result[0] += 0.016316103;
            }
          } else {
            result[0] += -0.003320934;
          }
        }
      } else {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
          result[0] += 0.053239834;
        } else {
          result[0] += 0.015470129;
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 64))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
              result[0] += -0.0021254455;
            } else {
              result[0] += -0.009888625;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
              result[0] += 0.062025126;
            } else {
              result[0] += -0.00554525;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 52))) {
              result[0] += 0.057511933;
            } else {
              result[0] += -0.008578778;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 52))) {
              result[0] += -0.046820056;
            } else {
              result[0] += -0.02672096;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 70))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
              result[0] += -0.03445236;
            } else {
              result[0] += 0.01575507;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
              result[0] += -0.029678077;
            } else {
              result[0] += -0.0074849683;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
              result[0] += 0.013723537;
            } else {
              result[0] += -0.0020312674;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
              result[0] += -0.004165123;
            } else {
              result[0] += -0.00078670814;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 12))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += -0;
            } else {
              result[0] += -0.03194522;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
              result[0] += 0.025147652;
            } else {
              result[0] += 0.0062306984;
            }
          }
        } else {
          result[0] += -0.0008928859;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
          result[0] += 0.13331777;
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 206))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 144))) {
              result[0] += 0.030818671;
            } else {
              result[0] += 0.011000312;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
              result[0] += 0.0052034413;
            } else {
              result[0] += -0.002728651;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.0031897593;
            } else {
              result[0] += 0.017273722;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
              result[0] += -0.02854721;
            } else {
              result[0] += 0.00037428603;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
              result[0] += -0.0019407201;
            } else {
              result[0] += -0.007221583;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 62))) {
              result[0] += 0.047634736;
            } else {
              result[0] += -0.017209576;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
            result[0] += 0.025280062;
          } else {
            result[0] += 0.012840076;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 44))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
              result[0] += 0.038194098;
            } else {
              result[0] += 0.008874618;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 46))) {
              result[0] += -0.03347216;
            } else {
              result[0] += -0.0004954495;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 50))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 172))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 202))) {
              result[0] += -0.009764884;
            } else {
              result[0] += -0.027206961;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 202))) {
              result[0] += 0.00784516;
            } else {
              result[0] += 0.0012178732;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 170))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 168))) {
              result[0] += 0.024105137;
            } else {
              result[0] += -0.0019981433;
            }
          } else {
            result[0] += 0.031363543;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
              result[0] += -0.0072048977;
            } else {
              result[0] += -0.022911293;
            }
          } else {
            result[0] += 0.010616104;
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 258))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 248))) {
              result[0] += 0.008845729;
            } else {
              result[0] += -0.0013868009;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 246))) {
              result[0] += -0.0024586073;
            } else {
              result[0] += -0.0086732535;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 196))) {
              result[0] += 0.01107409;
            } else {
              result[0] += -0.0011529119;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += 0.013766219;
            } else {
              result[0] += 0.026060035;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 210))) {
              result[0] += -0.004132199;
            } else {
              result[0] += -0.01266666;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 264))) {
              result[0] += 0.011719443;
            } else {
              result[0] += -0.0067507224;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 90))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 188))) {
            result[0] += 0.0155691;
          } else {
            result[0] += 0.010912938;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 198))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 188))) {
              result[0] += 0.0051367334;
            } else {
              result[0] += -0.00038397807;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.044876117;
            } else {
              result[0] += 0.005380819;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 124))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 250))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
              result[0] += -0.01634503;
            } else {
              result[0] += -0;
            }
          } else {
            result[0] += -0.002735744;
          }
        } else {
          result[0] += 0.012586943;
        }
      } else {
        result[0] += -0.0009229357;
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 106))) {
              result[0] += -0.013541499;
            } else {
              result[0] += 0.016169539;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += -0;
            } else {
              result[0] += 0.026296902;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 58))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
              result[0] += 0.0010251206;
            } else {
              result[0] += 0.007284665;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 100))) {
              result[0] += -0.0034584904;
            } else {
              result[0] += 0.0022243506;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 2))) {
              result[0] += 0.008507351;
            } else {
              result[0] += 0.00037087366;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 220))) {
              result[0] += -0.0015936692;
            } else {
              result[0] += -0.00584162;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.03250199;
            } else {
              result[0] += 0.014099632;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 174))) {
              result[0] += 0.0007955474;
            } else {
              result[0] += 0.0064386777;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 240))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 144))) {
              result[0] += -0.032964613;
            } else {
              result[0] += 0.017446639;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 134))) {
              result[0] += -0.015205093;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
              result[0] += 0.003139617;
            } else {
              result[0] += -0.014359586;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += -0.0014178219;
            } else {
              result[0] += 0.002840144;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 62))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
              result[0] += 0.041497182;
            } else {
              result[0] += 0.001743556;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
              result[0] += -0.014737419;
            } else {
              result[0] += -0.0060345526;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 188))) {
              result[0] += -0.0014039263;
            } else {
              result[0] += 0.009091877;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
              result[0] += -0.013735535;
            } else {
              result[0] += -0.0066661197;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 100))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += 0.05542008;
            } else {
              result[0] += 0.0056177597;
            }
          } else {
            result[0] += 0.021063056;
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
              result[0] += -0.008560927;
            } else {
              result[0] += 0.007372921;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
              result[0] += 0.0033190027;
            } else {
              result[0] += 0.02229972;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 162))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 272))) {
              result[0] += 0.0031475432;
            } else {
              result[0] += 0.01872689;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 42))) {
              result[0] += -0.0027235162;
            } else {
              result[0] += -0.011199118;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += -0.013897911;
            } else {
              result[0] += -0.0017742496;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 144))) {
              result[0] += 0.045032453;
            } else {
              result[0] += 0.0035365508;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
          result[0] += 0.015004215;
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
            result[0] += -0.02693589;
          } else {
            result[0] += 0.0030036382;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 236))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 220))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
              result[0] += -0.005196058;
            } else {
              result[0] += -0.0010462991;
            }
          } else {
            result[0] += -0.036202352;
          }
        } else {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
            result[0] += 0.009205636;
          } else {
            result[0] += -0.008914505;
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
              result[0] += 0.00077702984;
            } else {
              result[0] += -0.006057858;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 78))) {
              result[0] += 0.011342842;
            } else {
              result[0] += 0.003530657;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
            result[0] += 0.03748219;
          } else {
            result[0] += 0.014057839;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 154))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 102))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += -0.0016399727;
            } else {
              result[0] += -0.011802055;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.0016713021;
            } else {
              result[0] += 0.0056822044;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 34))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += 0.00386166;
            } else {
              result[0] += -0.0027170626;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 46))) {
              result[0] += 0.019643234;
            } else {
              result[0] += 0.0028665168;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 204))) {
            result[0] += 0.0005789458;
          } else {
            result[0] += 0.0143351955;
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 0))) {
              result[0] += 0.011239639;
            } else {
              result[0] += -0.016711237;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
              result[0] += 0.019091167;
            } else {
              result[0] += 0.0046307947;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            result[0] += 0.050568253;
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += 0.03797291;
            } else {
              result[0] += 0.007982335;
            }
          }
        } else {
          result[0] += 0.0035218038;
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
              result[0] += -0.016012253;
            } else {
              result[0] += 0.0023404376;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
              result[0] += -0.001683798;
            } else {
              result[0] += 0.013443462;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 232))) {
              result[0] += -0.0012087579;
            } else {
              result[0] += 0.009097977;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 192))) {
              result[0] += -0.010690723;
            } else {
              result[0] += -0.001824428;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            result[0] += 0.017058346;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 220))) {
              result[0] += 0.004044805;
            } else {
              result[0] += -0.0013507138;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 242))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
              result[0] += 0.008160069;
            } else {
              result[0] += 0.015147629;
            }
          } else {
            result[0] += 0.024462283;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
              result[0] += -0.000107747815;
            } else {
              result[0] += 0.0067256675;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
              result[0] += -0.0076294737;
            } else {
              result[0] += -0.043650094;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 178))) {
              result[0] += 0.034688678;
            } else {
              result[0] += -0.02312972;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 100))) {
              result[0] += -0.002360337;
            } else {
              result[0] += 0.012530643;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
          result[0] += -0.01826018;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 64))) {
              result[0] += 0.0056892973;
            } else {
              result[0] += 0.03400869;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 80))) {
              result[0] += -0.009709366;
            } else {
              result[0] += -0.00013415277;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 156))) {
              result[0] += 0.010154608;
            } else {
              result[0] += 0.001891119;
            }
          } else {
            result[0] += 0.03333105;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 154))) {
              result[0] += -0.022681896;
            } else {
              result[0] += -0.03837011;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 162))) {
              result[0] += 0.007119199;
            } else {
              result[0] += -0.015321928;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 134))) {
            result[0] += 0.08953688;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
              result[0] += -0.022155154;
            } else {
              result[0] += 0.004006593;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
              result[0] += 0.013605259;
            } else {
              result[0] += 0.027043974;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += -0.0115665;
            } else {
              result[0] += 0.007095266;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += 0.035634317;
            } else {
              result[0] += 0.017916704;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += -0.0043744547;
            } else {
              result[0] += -0.00026779916;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 236))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += 0.0133640645;
            } else {
              result[0] += 0.004636778;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += 0.012312482;
            } else {
              result[0] += 0.06577807;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 132))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.0068262727;
            } else {
              result[0] += -0.04068326;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
              result[0] += -0.0010328897;
            } else {
              result[0] += -0.005453972;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 146))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
              result[0] += -0.0008340759;
            } else {
              result[0] += -0.008702307;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 16))) {
              result[0] += 0.040576342;
            } else {
              result[0] += 0.0024016355;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 226))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 192))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += 0.029419197;
            } else {
              result[0] += 0.0010929945;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 156))) {
              result[0] += -0.0012289739;
            } else {
              result[0] += -0.0070381216;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
            result[0] += 0.0115409745;
          } else {
            result[0] += 0.0048038764;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 242))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 222))) {
            result[0] += 0.023705222;
          } else {
            result[0] += 0.010082607;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += 0.012382432;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 228))) {
              result[0] += 0.004983134;
            } else {
              result[0] += -0;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 208))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += 0.061853677;
            } else {
              result[0] += -0.0063631344;
            }
          } else {
            result[0] += -0.036711987;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 28))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
              result[0] += 0.0031359557;
            } else {
              result[0] += -0.0016080218;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
              result[0] += 0.036603495;
            } else {
              result[0] += 0.015373263;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.055269193;
            } else {
              result[0] += 0.019025035;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
              result[0] += -0.022278577;
            } else {
              result[0] += 0.016397122;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            result[0] += -0.01897105;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
              result[0] += 0.00051737565;
            } else {
              result[0] += 0.0041289385;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
          result[0] += -0.036939483;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
              result[0] += 0.009198739;
            } else {
              result[0] += -0.013264345;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.022775693;
            } else {
              result[0] += 0.007945242;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 156))) {
              result[0] += -0.0018210205;
            } else {
              result[0] += 0.0037362184;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 128))) {
              result[0] += -0.012762427;
            } else {
              result[0] += -0.0007420523;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 218))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
              result[0] += 0.017958501;
            } else {
              result[0] += 0.0061608446;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 88))) {
              result[0] += -0.0049462705;
            } else {
              result[0] += 0.0019659882;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
          result[0] += 0.00022622973;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 250))) {
            result[0] += 0.005413284;
          } else {
            result[0] += 0.030752031;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 220))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
            result[0] += 0.003555048;
          } else {
            result[0] += -0.0069270222;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
              result[0] += -0.008863601;
            } else {
              result[0] += -0.015783666;
            }
          } else {
            result[0] += -0.0067091286;
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 36))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 58))) {
          result[0] += 0.062294442;
        } else {
          result[0] += 0.01808921;
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
          result[0] += 0.08561305;
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 190))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += 0.0024828026;
            } else {
              result[0] += 0.018881498;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 204))) {
              result[0] += -0.007572981;
            } else {
              result[0] += 0.001793948;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 126))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
              result[0] += -0.000932767;
            } else {
              result[0] += 0.006823505;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
              result[0] += -0.036651295;
            } else {
              result[0] += -0.0058056363;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
              result[0] += -0.0055914307;
            } else {
              result[0] += 0.03341748;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 100))) {
              result[0] += -0.00073259696;
            } else {
              result[0] += 0.010309592;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
          result[0] += -0.019160004;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 64))) {
              result[0] += 0.0046965308;
            } else {
              result[0] += 0.026486978;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 80))) {
              result[0] += -0.008773739;
            } else {
              result[0] += -0.0007050743;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
            result[0] += 0.025857493;
          } else {
            result[0] += -0.0017104832;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 46))) {
            result[0] += -0.04572242;
          } else {
            result[0] += -0.013417631;
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 128))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += 0.02735805;
            } else {
              result[0] += 0.008384274;
            }
          } else {
            result[0] += 0.025471484;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 146))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 144))) {
              result[0] += -0.0019103719;
            } else {
              result[0] += -0.015735812;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
              result[0] += 0.003526498;
            } else {
              result[0] += 0.0004870697;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 252))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 230))) {
              result[0] += -0.0010278291;
            } else {
              result[0] += 0.0035127073;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
              result[0] += 0.0007956819;
            } else {
              result[0] += -0.007078775;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
              result[0] += 0.06692124;
            } else {
              result[0] += 0.012191664;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
              result[0] += -0.0037893832;
            } else {
              result[0] += 0.0049554044;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 262))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            result[0] += -0.0026184532;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
              result[0] += -0.03463073;
            } else {
              result[0] += -0.01464231;
            }
          }
        } else {
          result[0] += 0.01621621;
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
          result[0] += -0.0019204362;
        } else {
          result[0] += 0.012864575;
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
          result[0] += -0.008148334;
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 46))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 212))) {
              result[0] += 0.04631585;
            } else {
              result[0] += 0.008748725;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 58))) {
              result[0] += -0.0012717935;
            } else {
              result[0] += 0.0031659238;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
        result[0] += 0.023349551;
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
            result[0] += 0.0070865126;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 220))) {
              result[0] += -0.0010378936;
            } else {
              result[0] += -0.03451775;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 58))) {
              result[0] += -0.0030549995;
            } else {
              result[0] += 0.042464044;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 196))) {
              result[0] += -0.0068810317;
            } else {
              result[0] += -0.01983464;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 28))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
              result[0] += 0.021494946;
            } else {
              result[0] += 0.046589624;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 246))) {
              result[0] += 0.0016733175;
            } else {
              result[0] += -0.002286189;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
              result[0] += -0.023560746;
            } else {
              result[0] += 0.025176898;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
              result[0] += -0.0032716715;
            } else {
              result[0] += -0.0107773235;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 18))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 158))) {
              result[0] += 9.897955e-07;
            } else {
              result[0] += 0.0075915637;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 238))) {
              result[0] += -0.018133821;
            } else {
              result[0] += -0.003638253;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
              result[0] += 0.013451889;
            } else {
              result[0] += 0.006885908;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 2))) {
              result[0] += 0.016480822;
            } else {
              result[0] += 0.0028402929;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 182))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
              result[0] += 0.009219227;
            } else {
              result[0] += -0.025390211;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.025732726;
            } else {
              result[0] += -0.0023830726;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 204))) {
              result[0] += 0.0035509118;
            } else {
              result[0] += 0.013229172;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.007854169;
            } else {
              result[0] += 0.0072692037;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            result[0] += 0.047711357;
          } else {
            result[0] += 0.025087982;
          }
        } else {
          result[0] += 0.0016552185;
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 36))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            result[0] += 0.0398016;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
              result[0] += 0.0008522939;
            } else {
              result[0] += 0.022818817;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 8))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 180))) {
              result[0] += -0.047148183;
            } else {
              result[0] += -0.014362188;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
              result[0] += 0.033707537;
            } else {
              result[0] += -0.005045968;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 38))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 52))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
              result[0] += -0.015981285;
            } else {
              result[0] += 0.061333533;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.011986034;
            } else {
              result[0] += 0.014650764;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 156))) {
              result[0] += 0.03378779;
            } else {
              result[0] += 0.001349721;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += -0.00046623495;
            } else {
              result[0] += -0.0032709301;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 64))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 42))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.0026227154;
            } else {
              result[0] += -0.004871029;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.024236446;
            } else {
              result[0] += 0.00875364;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 86))) {
              result[0] += -0.010391722;
            } else {
              result[0] += -0.0003864034;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 58))) {
              result[0] += 0.0091726575;
            } else {
              result[0] += 0.00034549323;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += -0.016140882;
            } else {
              result[0] += -0.0051438985;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 86))) {
              result[0] += 0.0077548027;
            } else {
              result[0] += -0.0023856275;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 96))) {
            result[0] += 0.0072452063;
          } else {
            result[0] += 0.021926673;
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
              result[0] += 0.007803672;
            } else {
              result[0] += 0.07842522;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
              result[0] += 0.0030498195;
            } else {
              result[0] += -0.030465296;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 138))) {
            result[0] += 0.027338846;
          } else {
            result[0] += 0.018519042;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 104))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 150))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
              result[0] += -0.0038131357;
            } else {
              result[0] += -0.014322984;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
              result[0] += 0.011482879;
            } else {
              result[0] += -0.0018702493;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 38))) {
            result[0] += 0.04208499;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 48))) {
              result[0] += -0.013598467;
            } else {
              result[0] += 0.0022218376;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += 0.014965734;
            } else {
              result[0] += 0.006420965;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
              result[0] += 0.0035459124;
            } else {
              result[0] += -0.03692886;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            result[0] += -0.00581108;
          } else {
            result[0] += 0.0022818777;
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          result[0] += 0.02711085;
        } else {
          result[0] += 0.0037428073;
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 232))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 230))) {
              result[0] += -0.0006429512;
            } else {
              result[0] += -0.013558619;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
              result[0] += 0.006967464;
            } else {
              result[0] += 0.020120773;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
              result[0] += -0.0188818;
            } else {
              result[0] += -0.011276356;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 248))) {
              result[0] += -0.0036909913;
            } else {
              result[0] += 0.0023398104;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 228))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 220))) {
            result[0] += 0.0067386203;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
              result[0] += 0.009782809;
            } else {
              result[0] += -0.0003835666;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 242))) {
            result[0] += 0.008386671;
          } else {
            result[0] += 0.01509656;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 230))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += 0.0035304844;
            } else {
              result[0] += 0.00860195;
            }
          } else {
            result[0] += 0.044866074;
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
              result[0] += -0.0027610434;
            } else {
              result[0] += 0.032996777;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 30))) {
              result[0] += 0.009394963;
            } else {
              result[0] += -0.0004232897;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
              result[0] += 0.019008147;
            } else {
              result[0] += 0.009637523;
            }
          } else {
            result[0] += 0.041156612;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
            result[0] += 0.0076490478;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 182))) {
              result[0] += 0.01141602;
            } else {
              result[0] += 0.0032564194;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 4))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
              result[0] += 0.0054725604;
            } else {
              result[0] += -0.006845514;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
              result[0] += -0;
            } else {
              result[0] += 0.04772381;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += -0.016645087;
            } else {
              result[0] += -0.032690536;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
              result[0] += 0.003037467;
            } else {
              result[0] += -0.009953394;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 12))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
            result[0] += 0.01011019;
          } else {
            result[0] += 0.017591367;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 258))) {
              result[0] += 0.0020601144;
            } else {
              result[0] += -0.0052700583;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 172))) {
              result[0] += -0.0001318003;
            } else {
              result[0] += -0.0058084927;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 160))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 158))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 8))) {
              result[0] += -0.021778304;
            } else {
              result[0] += -0.05481277;
            }
          } else {
            result[0] += 0.01922178;
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 180))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += -0.02884279;
            } else {
              result[0] += -0.05432384;
            }
          } else {
            result[0] += -0.008790418;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
            result[0] += -0.0039755395;
          } else {
            result[0] += 0.0508219;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 128))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 28))) {
              result[0] += -0.047565266;
            } else {
              result[0] += -0.034321543;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += 0.011836532;
            } else {
              result[0] += -0.0025003483;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 58))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 68))) {
              result[0] += 0.0057843057;
            } else {
              result[0] += 0.036940213;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
              result[0] += -0.011010564;
            } else {
              result[0] += 0.014086082;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 226))) {
              result[0] += -0.00865682;
            } else {
              result[0] += 0.018309247;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 218))) {
              result[0] += 0.00883958;
            } else {
              result[0] += -0.0026359025;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 98))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 94))) {
              result[0] += -0.0024397147;
            } else {
              result[0] += -0.023152852;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
              result[0] += 0.0015076617;
            } else {
              result[0] += -0.009175034;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += 0.02444943;
            } else {
              result[0] += 0.0023199369;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
              result[0] += -0.0028377601;
            } else {
              result[0] += 0.00046475232;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 64))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += 0.00531006;
            } else {
              result[0] += -0.00012114655;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
              result[0] += -0.004693984;
            } else {
              result[0] += -0.032999605;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.029490167;
            } else {
              result[0] += 0.007921627;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 58))) {
              result[0] += -0.0028598853;
            } else {
              result[0] += 0.0077966005;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 66))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 70))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
              result[0] += 0.014692311;
            } else {
              result[0] += -0.021459384;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 94))) {
              result[0] += -0.016421737;
            } else {
              result[0] += -0.0066403346;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += 0.0074493513;
            } else {
              result[0] += -0.0040894877;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
              result[0] += 0.01759093;
            } else {
              result[0] += -0.00061179395;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 18))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
            result[0] += 0.0076502063;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 236))) {
              result[0] += 0.00010772177;
            } else {
              result[0] += -0.003537047;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
            result[0] += -0.012459938;
          } else {
            result[0] += -0.0037218079;
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 260))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += 0.002977729;
            } else {
              result[0] += -0.019293662;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
              result[0] += 0.015566354;
            } else {
              result[0] += 0.0037241564;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            result[0] += 0.03374357;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += 0.0041862135;
            } else {
              result[0] += 0.009111217;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 204))) {
            result[0] += 0.001820308;
          } else {
            result[0] += 0.0113331815;
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
              result[0] += -0.01062336;
            } else {
              result[0] += 0.0023687228;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
              result[0] += 0.0050820797;
            } else {
              result[0] += -0.010689573;
            }
          }
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            result[0] += 0.039545674;
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += 0.023330238;
            } else {
              result[0] += 0.0062547647;
            }
          }
        } else {
          result[0] += 0.00083993434;
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 58))) {
              result[0] += -0.0019025843;
            } else {
              result[0] += 0.0075600627;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
              result[0] += 0.042078517;
            } else {
              result[0] += 0.0026668767;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
            result[0] += -0.015797917;
          } else {
            result[0] += -0.006350769;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 96))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
            result[0] += 0.048628982;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
              result[0] += 0.021388667;
            } else {
              result[0] += 0.0020165527;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 110))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 106))) {
              result[0] += -0.0051917243;
            } else {
              result[0] += 0.010212748;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += 0.0015873375;
            } else {
              result[0] += -0.0008199594;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 64))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 212))) {
              result[0] += 0.01901365;
            } else {
              result[0] += -0.0028177812;
            }
          } else {
            result[0] += 0.02409505;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 206))) {
              result[0] += -0.0063365176;
            } else {
              result[0] += 0.00086170074;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
              result[0] += -0.016780075;
            } else {
              result[0] += 0.012628958;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
          result[0] += -0.009542297;
        } else {
          result[0] += 0.053097267;
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 250))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 238))) {
              result[0] += 0.003721561;
            } else {
              result[0] += 0.019598227;
            }
          } else {
            result[0] += 0.03498596;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 68))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
              result[0] += 0.00025514522;
            } else {
              result[0] += 0.0040203123;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
              result[0] += -0.0074286456;
            } else {
              result[0] += 0.00093450863;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 246))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 166))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
              result[0] += -0.00087522157;
            } else {
              result[0] += 0.007016174;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
              result[0] += -0.034118887;
            } else {
              result[0] += -0.011294309;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 260))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 256))) {
              result[0] += 0.022198912;
            } else {
              result[0] += 0.012400496;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
              result[0] += -0.010423508;
            } else {
              result[0] += 0.004261658;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 96))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 52))) {
              result[0] += -0.001626932;
            } else {
              result[0] += 0.0061591933;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
              result[0] += -0.0022012205;
            } else {
              result[0] += -0.0123000285;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 144))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 122))) {
              result[0] += 0.007472606;
            } else {
              result[0] += 0.05678388;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 196))) {
              result[0] += 3.3709428e-05;
            } else {
              result[0] += -0.0143476175;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 50))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
              result[0] += 0.03503921;
            } else {
              result[0] += -0.036016237;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 182))) {
              result[0] += -0.0048608086;
            } else {
              result[0] += 0.022741511;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            result[0] += -0.035393443;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 88))) {
              result[0] += -0.002815163;
            } else {
              result[0] += -2.7660133e-05;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 174))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 158))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 148))) {
              result[0] += 0.0017332671;
            } else {
              result[0] += 0.026441107;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 222))) {
              result[0] += -0.0073182597;
            } else {
              result[0] += 0.0006405428;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
            result[0] += 0.02126172;
          } else {
            result[0] += 0.011408723;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 68))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 46))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
              result[0] += -0.014773398;
            } else {
              result[0] += 0.009008159;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.0047272076;
            } else {
              result[0] += -0.01409996;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 160))) {
              result[0] += 0.009959228;
            } else {
              result[0] += 0.04201186;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
              result[0] += -0.008849121;
            } else {
              result[0] += -0.00033711825;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 16))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
              result[0] += 0.002547614;
            } else {
              result[0] += -0.016982568;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
              result[0] += 0.0087383045;
            } else {
              result[0] += -0.0019732136;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
              result[0] += -0.027964113;
            } else {
              result[0] += -0.0070363963;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 12))) {
              result[0] += 0.015301444;
            } else {
              result[0] += -0.0009437213;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 218))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 22))) {
              result[0] += 0.011154254;
            } else {
              result[0] += 0.03460535;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
              result[0] += 0.006955503;
            } else {
              result[0] += 0.002656106;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 30))) {
            result[0] += 0.008213889;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 234))) {
              result[0] += -0.0014978464;
            } else {
              result[0] += 0.0018082943;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 166))) {
              result[0] += -0.010998022;
            } else {
              result[0] += -0.028136352;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 28))) {
              result[0] += 0.056712236;
            } else {
              result[0] += 0.00025265655;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
            result[0] += -0.032531813;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.012461627;
            } else {
              result[0] += -0.0026103533;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 62))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
            result[0] += 0.003723057;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 44))) {
              result[0] += 0.050172295;
            } else {
              result[0] += 0.03318845;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += 0.038616996;
            } else {
              result[0] += -0.010049296;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 32))) {
              result[0] += -0.0062160105;
            } else {
              result[0] += -4.625852e-05;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 228))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 216))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 220))) {
              result[0] += 0.027368566;
            } else {
              result[0] += -0.0007781659;
            }
          } else {
            result[0] += -0.010964031;
          }
        } else {
          result[0] += 0.0031334688;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 286))) {
            result[0] += 0.023848334;
          } else {
            result[0] += 0.008595723;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
            result[0] += 0.0113761;
          } else {
            result[0] += 0.0038027503;
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
          result[0] += 0.028975934;
        } else {
          result[0] += 0.01324735;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
            result[0] += -0;
          } else {
            result[0] += 0.011437596;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
            result[0] += 0.01134281;
          } else {
            result[0] += 0.0018215694;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 242))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 230))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 114))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 106))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 180))) {
              result[0] += 0.0014854757;
            } else {
              result[0] += -0.0047508436;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
              result[0] += -0.012526001;
            } else {
              result[0] += -0.0042038257;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += 0.027467722;
            } else {
              result[0] += 0.005847614;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
              result[0] += -0.016319519;
            } else {
              result[0] += 0.0018143759;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 138))) {
            result[0] += 0.0017114735;
          } else {
            result[0] += -0.04394586;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.014061573;
            } else {
              result[0] += 0.0076026404;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 184))) {
              result[0] += -0.0018655244;
            } else {
              result[0] += 0.011241974;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 54))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
          result[0] += -0.013273408;
        } else {
          result[0] += 0.015772188;
        }
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 90))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 58))) {
              result[0] += -0.00058818224;
            } else {
              result[0] += 0.008676103;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
              result[0] += -0.010151858;
            } else {
              result[0] += 0.001280522;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 184))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
              result[0] += -0.010351776;
            } else {
              result[0] += 0.012125977;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
              result[0] += -0.005976769;
            } else {
              result[0] += 0.0058461004;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 22))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
              result[0] += 0.0014196447;
            } else {
              result[0] += -0.016857633;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 18))) {
              result[0] += 0.0388976;
            } else {
              result[0] += 0.0022651523;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 46))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
              result[0] += -0.0031505611;
            } else {
              result[0] += -0.02737777;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
              result[0] += 0.027305475;
            } else {
              result[0] += -0.00032103845;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 28))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
            result[0] += -0.033979844;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 44))) {
              result[0] += -0.00036602697;
            } else {
              result[0] += -0.009034148;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 44))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
              result[0] += 0.02605836;
            } else {
              result[0] += 0.0035190487;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
              result[0] += 0.0030745373;
            } else {
              result[0] += 0.021341039;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 112))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += -0.00765642;
            } else {
              result[0] += 0.012637125;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 92))) {
              result[0] += -0.031098261;
            } else {
              result[0] += -0.018325815;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 48))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 74))) {
              result[0] += 0.007115057;
            } else {
              result[0] += 0.025569752;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 50))) {
              result[0] += -0.030833412;
            } else {
              result[0] += -0.0027387291;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 118))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
            result[0] += -0.010184347;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
              result[0] += 0.009563519;
            } else {
              result[0] += 0.0019127353;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
              result[0] += -0.0016026145;
            } else {
              result[0] += -0.012827705;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
              result[0] += 0.006836459;
            } else {
              result[0] += -0.00047072658;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
      result[0] += 0.016250825;
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
          result[0] += -0.009732073;
        } else {
          result[0] += 0.017143272;
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
          result[0] += -0.009671597;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 270))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 134))) {
              result[0] += -0.00035089705;
            } else {
              result[0] += 0.004113893;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
              result[0] += 0.01820572;
            } else {
              result[0] += 0.009231458;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
        result[0] += 0.00027064016;
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.0017144865;
            } else {
              result[0] += -0.0095933005;
            }
          } else {
            result[0] += -0.012053235;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 194))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 250))) {
              result[0] += -0.0034071447;
            } else {
              result[0] += 0.046097945;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 132))) {
              result[0] += -0.0055225524;
            } else {
              result[0] += -0.0019266952;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 250))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 86))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
              result[0] += 0.0048346436;
            } else {
              result[0] += 0.01435117;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
              result[0] += -0.019338546;
            } else {
              result[0] += -0.0010408615;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 184))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 178))) {
              result[0] += 0.011773284;
            } else {
              result[0] += 0.002931762;
            }
          } else {
            result[0] += 0.040974855;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 146))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
              result[0] += 0.08329225;
            } else {
              result[0] += -0.0002780066;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
              result[0] += -0.0042001484;
            } else {
              result[0] += -0.02084507;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.00010066136;
            } else {
              result[0] += 0.0027292925;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += -0.0122973155;
            } else {
              result[0] += 0.035760295;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
            result[0] += -0.001996606;
          } else {
            result[0] += 0.021067668;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 118))) {
              result[0] += -0.013935153;
            } else {
              result[0] += 0.031183664;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += 0.061950255;
            } else {
              result[0] += 0.0058407323;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 142))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 72))) {
            result[0] += 0.0829713;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.03793059;
            } else {
              result[0] += -0.00907617;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 68))) {
              result[0] += 0.0025514578;
            } else {
              result[0] += -0.0012025639;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 16))) {
              result[0] += 0.046701174;
            } else {
              result[0] += 0.0046325247;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 148))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 130))) {
            result[0] += 0.027825123;
          } else {
            result[0] += -0;
          }
        } else {
          result[0] += -0.021053893;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 82))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 152))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 106))) {
              result[0] += -0.0013631139;
            } else {
              result[0] += -0.012569079;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
              result[0] += -0.0008349316;
            } else {
              result[0] += 0.022382598;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 156))) {
              result[0] += 0.013735485;
            } else {
              result[0] += 0.0033138485;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
              result[0] += -0.0037839809;
            } else {
              result[0] += 0.002062925;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 22))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
            result[0] += 0.0013772444;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
              result[0] += -0.011463809;
            } else {
              result[0] += -0.0047910134;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
              result[0] += 0.060537297;
            } else {
              result[0] += 0.010533939;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.016672423;
            } else {
              result[0] += -0.00052434875;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 250))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
              result[0] += 0.0070519983;
            } else {
              result[0] += 0.016810419;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
              result[0] += -0.026641011;
            } else {
              result[0] += 0.00010854311;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 232))) {
            result[0] += 0.03818845;
          } else {
            result[0] += 0.012029546;
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
          result[0] += -0.007278714;
        } else {
          result[0] += -0.03333238;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 144))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 46))) {
              result[0] += 0.007880951;
            } else {
              result[0] += -0.00031094736;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += 0.013625564;
            } else {
              result[0] += 0.03885763;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 214))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
              result[0] += -0.0029387611;
            } else {
              result[0] += 7.662349e-05;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.029973779;
            } else {
              result[0] += 0.0043506036;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 146))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 78))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
              result[0] += -0.017686712;
            } else {
              result[0] += 0.015614047;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 242))) {
              result[0] += 0.0010816348;
            } else {
              result[0] += 0.010603439;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 122))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += -0.013882413;
            } else {
              result[0] += -3.773386e-06;
            }
          } else {
            result[0] += -0.029591149;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 52))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
              result[0] += 0.0029416822;
            } else {
              result[0] += 0.060454708;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 52))) {
              result[0] += -0.03178696;
            } else {
              result[0] += -0.011132187;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 164))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
              result[0] += 0.06373953;
            } else {
              result[0] += -0.0049581574;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
              result[0] += -0.011388632;
            } else {
              result[0] += 0.0049218778;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 14))) {
              result[0] += 0.0012467568;
            } else {
              result[0] += -0.019324591;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 44))) {
              result[0] += 0.001289832;
            } else {
              result[0] += -0.00096984406;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 44))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
              result[0] += -0.011180186;
            } else {
              result[0] += -0.0001193571;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += 0.017248346;
            } else {
              result[0] += 0.003171424;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 18))) {
          result[0] += 0.04394267;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 176))) {
            result[0] += -0.026358157;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += 0.0084524695;
            } else {
              result[0] += -0.008720714;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 286))) {
      result[0] += 0.02462693;
    } else {
      result[0] += 0.0102696465;
    }
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 236))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 190))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
              result[0] += -9.075111e-05;
            } else {
              result[0] += -0.0032513805;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 202))) {
              result[0] += 0.0011170026;
            } else {
              result[0] += 0.023266837;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 204))) {
              result[0] += -0.012011125;
            } else {
              result[0] += -0.0051451973;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 202))) {
              result[0] += 0.023120182;
            } else {
              result[0] += -0.008411562;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 212))) {
          result[0] += 0.017263098;
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 232))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 230))) {
              result[0] += 0.003063401;
            } else {
              result[0] += 0.021211077;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 234))) {
              result[0] += -0.0117486585;
            } else {
              result[0] += -0.0008188842;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 248))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 40))) {
              result[0] += -0.025011426;
            } else {
              result[0] += -0.01188937;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 90))) {
              result[0] += 0.000581751;
            } else {
              result[0] += 0.035827655;
            }
          }
        } else {
          result[0] += -0.056811877;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
          result[0] += 0.03448122;
        } else {
          result[0] += 0.0075843707;
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 120))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 150))) {
            result[0] += 0.02317158;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
              result[0] += -0.011207158;
            } else {
              result[0] += 0.00603795;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
              result[0] += 1.6565642e-05;
            } else {
              result[0] += 0.0018936057;
            }
          } else {
            result[0] += -0.0073121726;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
          result[0] += 0.013665982;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
            result[0] += 0.02345163;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 90))) {
              result[0] += 0.031935133;
            } else {
              result[0] += 0.05211563;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 108))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 150))) {
              result[0] += 0.006929166;
            } else {
              result[0] += 0.0018365175;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
              result[0] += -0.028940136;
            } else {
              result[0] += -0.0025783076;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.010843903;
            } else {
              result[0] += 0.006668359;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 84))) {
              result[0] += -0.022552337;
            } else {
              result[0] += -0.0126125915;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 188))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
              result[0] += -0.0033015904;
            } else {
              result[0] += 0.0023345288;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 172))) {
              result[0] += 0.027362315;
            } else {
              result[0] += 0.008473503;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 198))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.0086487485;
            } else {
              result[0] += 0.0011204499;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
              result[0] += -0.0040217172;
            } else {
              result[0] += 0.0013360606;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 202))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
              result[0] += -0.0038944704;
            } else {
              result[0] += -0.024639545;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
              result[0] += 0.010567961;
            } else {
              result[0] += -0.0008910143;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 222))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 208))) {
              result[0] += 0.0022678128;
            } else {
              result[0] += 0.010532487;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 20))) {
              result[0] += 0.0032915582;
            } else {
              result[0] += -0.0005411969;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 152))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += -0.01407851;
            } else {
              result[0] += 0.026896363;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 42))) {
              result[0] += -0.007195405;
            } else {
              result[0] += 0.004968326;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 174))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
              result[0] += 0.008754817;
            } else {
              result[0] += -0.0017400451;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += -0.00865253;
            } else {
              result[0] += 0.0009031243;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 244))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
              result[0] += 0.00038980006;
            } else {
              result[0] += -0.016266255;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 18))) {
              result[0] += -0.011384182;
            } else {
              result[0] += -0.005449654;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
            result[0] += 0.029928738;
          } else {
            result[0] += -0.005479242;
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 50))) {
          result[0] += -0.02587893;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 182))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
              result[0] += -0.005359716;
            } else {
              result[0] += 0.06967212;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 236))) {
              result[0] += 0.0018839792;
            } else {
              result[0] += 0.00916981;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 122))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 104))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += 0.0006954739;
            } else {
              result[0] += 0.048395343;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
              result[0] += 1.917278e-05;
            } else {
              result[0] += 0.03755577;
            }
          }
        } else {
          result[0] += -0.011912975;
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 118))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 220))) {
              result[0] += 0.00019196502;
            } else {
              result[0] += 0.014448823;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
              result[0] += 0.019970976;
            } else {
              result[0] += -0.0011340084;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
              result[0] += -0.009018285;
            } else {
              result[0] += 0.020422988;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
              result[0] += 0.02001476;
            } else {
              result[0] += 0.034766663;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 152))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += -0.0060089673;
            } else {
              result[0] += -0.026388485;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 102))) {
              result[0] += -0.0338698;
            } else {
              result[0] += -0.0058778613;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 192))) {
            result[0] += 0.025475418;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += 0.0019010755;
            } else {
              result[0] += 0.011219644;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 128))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
              result[0] += -0.021041693;
            } else {
              result[0] += 0.008361343;
            }
          } else {
            result[0] += 0.03766183;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
              result[0] += -0.010491718;
            } else {
              result[0] += 0.003890305;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.011144113;
            } else {
              result[0] += -8.4675696e-05;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 212))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 166))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
              result[0] += 0.001330308;
            } else {
              result[0] += 0.02066918;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 130))) {
              result[0] += 6.2536885e-05;
            } else {
              result[0] += -0.013168806;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 90))) {
              result[0] += 0.019082962;
            } else {
              result[0] += 0.0053747944;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
              result[0] += -0.0076252148;
            } else {
              result[0] += 0.0009393127;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 168))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 94))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.011447157;
            } else {
              result[0] += 0.0061047096;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
              result[0] += -0;
            } else {
              result[0] += -0.020285398;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.01896581;
            } else {
              result[0] += 0.00095191033;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 172))) {
              result[0] += 0.0154045075;
            } else {
              result[0] += -0.001370645;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
            result[0] += 0.006603177;
          } else {
            result[0] += 0.00034456272;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
            result[0] += -0.0029918782;
          } else {
            result[0] += -0.012741557;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 166))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 164))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
              result[0] += 0.0071618394;
            } else {
              result[0] += -0.012161553;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += 0.03479741;
            } else {
              result[0] += 0.00893333;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 178))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 228))) {
              result[0] += 0.0005804447;
            } else {
              result[0] += -0.006457042;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 186))) {
              result[0] += 0.016656972;
            } else {
              result[0] += 0.002414147;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 116))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 20))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 124))) {
              result[0] += 0.022543216;
            } else {
              result[0] += 0.0024742563;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 50))) {
              result[0] += -0.025850767;
            } else {
              result[0] += -0.0073662284;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 46))) {
              result[0] += 0.0006718464;
            } else {
              result[0] += -0.0011177754;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 110))) {
              result[0] += 0.003656535;
            } else {
              result[0] += -0.0034576224;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 122))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
              result[0] += 0.01316909;
            } else {
              result[0] += -0.00022220907;
            }
          } else {
            result[0] += 0.0322718;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 132))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += 0.0029699726;
            } else {
              result[0] += -0.009400754;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
              result[0] += 0.0157076;
            } else {
              result[0] += -0.0039490475;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 136))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 192))) {
            result[0] += -0.030938972;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 174))) {
              result[0] += 0.027307421;
            } else {
              result[0] += -0.00040336605;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
            result[0] += 0.029921725;
          } else {
            result[0] += 0.013266409;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 168))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 62))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 144))) {
              result[0] += 0.044555373;
            } else {
              result[0] += 0.0020996393;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 90))) {
              result[0] += -0.0010202875;
            } else {
              result[0] += -0.0097485995;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 170))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
              result[0] += 0.024518076;
            } else {
              result[0] += -0.0032617298;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += -0.002727569;
            } else {
              result[0] += -0.00034776315;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 188))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
              result[0] += -0.00013192062;
            } else {
              result[0] += -0.007137733;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
              result[0] += 0.0058402964;
            } else {
              result[0] += 0.00019643598;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 170))) {
            result[0] += -0.015256891;
          } else {
            result[0] += 0.027069787;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
              result[0] += 0.016238226;
            } else {
              result[0] += 0.045753922;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
              result[0] += -0.019134747;
            } else {
              result[0] += 0.010413033;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 178))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 122))) {
              result[0] += -0.0016630454;
            } else {
              result[0] += -0.024183424;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
              result[0] += 0.012589683;
            } else {
              result[0] += 0.0009457501;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 210))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 208))) {
              result[0] += -0.0066352724;
            } else {
              result[0] += -0.017913656;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.0011060528;
            } else {
              result[0] += 0.0039828527;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 200))) {
              result[0] += -0.0031051391;
            } else {
              result[0] += -0.016724786;
            }
          } else {
            result[0] += -0.024875712;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 162))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 72))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 116))) {
              result[0] += -0.016162679;
            } else {
              result[0] += 0.0026748257;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 160))) {
              result[0] += 0.01718258;
            } else {
              result[0] += 0.0035327424;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 190))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 114))) {
              result[0] += 0.0027622434;
            } else {
              result[0] += -0.019732043;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
              result[0] += -0.009568975;
            } else {
              result[0] += -0.0024784;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 212))) {
      result[0] += 0.01407246;
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
              result[0] += -0.04716774;
            } else {
              result[0] += -0.008298879;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
              result[0] += -0.012182956;
            } else {
              result[0] += -0.0051582577;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 124))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 128))) {
              result[0] += -0.0067259143;
            } else {
              result[0] += 0.013526896;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
              result[0] += 0.002809637;
            } else {
              result[0] += -0.0016421535;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 86))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 82))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += -0.0011062829;
            } else {
              result[0] += 0.014348181;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
              result[0] += 0.0076159174;
            } else {
              result[0] += -0.0151206255;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 124))) {
              result[0] += 0.0045484654;
            } else {
              result[0] += 0.015772928;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
              result[0] += -0.0059127775;
            } else {
              result[0] += 0.0025872302;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 206))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
        result[0] += 0.115350716;
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
          result[0] += 0.030523414;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
              result[0] += 0.00818327;
            } else {
              result[0] += 0.042636707;
            }
          } else {
            result[0] += 0.002394322;
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += -0;
            } else {
              result[0] += 0.022605298;
            }
          } else {
            result[0] += -0.018712113;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
              result[0] += -0.018390236;
            } else {
              result[0] += 0.038224217;
            }
          } else {
            result[0] += 0.0023876673;
          }
        }
      } else {
        result[0] += -0.0001913241;
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 182))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
              result[0] += 0.002910438;
            } else {
              result[0] += -0.036367122;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
              result[0] += 0.0029667527;
            } else {
              result[0] += 0.04206534;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.012119245;
            } else {
              result[0] += -0.023951335;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 178))) {
              result[0] += -0.034575075;
            } else {
              result[0] += 0.007937021;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 138))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
              result[0] += -0.0042928117;
            } else {
              result[0] += -0.0012923562;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 184))) {
              result[0] += -0.028960615;
            } else {
              result[0] += -0.016311336;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.053593513;
            } else {
              result[0] += 0.009981692;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 78))) {
              result[0] += -0.017819297;
            } else {
              result[0] += -0.00070253096;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 200))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 44))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 32))) {
              result[0] += 0.004783728;
            } else {
              result[0] += 0.016091399;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 24))) {
              result[0] += 0.029074514;
            } else {
              result[0] += 0.002160437;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 250))) {
            result[0] += 0.055964243;
          } else {
            result[0] += 0.01622332;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 250))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 20))) {
              result[0] += -0.008321747;
            } else {
              result[0] += -0.0040490143;
            }
          } else {
            result[0] += -0.018693618;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 250))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += 6.4203756e-05;
            } else {
              result[0] += -0.0021725388;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 268))) {
              result[0] += 0.0055606547;
            } else {
              result[0] += -0.0023256661;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 182))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
              result[0] += -6.4768305e-06;
            } else {
              result[0] += -0.013116406;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.019466847;
            } else {
              result[0] += 0.006013643;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 178))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += -0.026702961;
            } else {
              result[0] += 0.0020189236;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 228))) {
              result[0] += -0.0006026015;
            } else {
              result[0] += 0.01105925;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
            result[0] += 0.029843587;
          } else {
            result[0] += 0.0661096;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 98))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 100))) {
              result[0] += 0.00522026;
            } else {
              result[0] += -0.0006855772;
            }
          } else {
            result[0] += 0.013178887;
          }
        }
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 124))) {
              result[0] += -0.020214727;
            } else {
              result[0] += 0.0038588562;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 200))) {
              result[0] += -0.00505647;
            } else {
              result[0] += -0.026435608;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 270))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
              result[0] += -0.0028923058;
            } else {
              result[0] += -0.008900782;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
              result[0] += 0.050368883;
            } else {
              result[0] += 0.0024831446;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 46))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 42))) {
              result[0] += 0.011402026;
            } else {
              result[0] += -0.031253975;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 190))) {
              result[0] += -0.012896495;
            } else {
              result[0] += -0.0015336399;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 68))) {
            result[0] += 0.011413305;
          } else {
            result[0] += 0.02147497;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 212))) {
      result[0] += 0.013494618;
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 46))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 260))) {
              result[0] += 0.0066242903;
            } else {
              result[0] += 0.0001638933;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
              result[0] += -0.009525633;
            } else {
              result[0] += 0.0022819052;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 176))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 70))) {
              result[0] += 0.0022807613;
            } else {
              result[0] += -0.0062286295;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
              result[0] += -0.01628782;
            } else {
              result[0] += -0.004168268;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 76))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 104))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 102))) {
              result[0] += 0.011599141;
            } else {
              result[0] += 0.0024420898;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 162))) {
              result[0] += 0.013803795;
            } else {
              result[0] += -0.0013663528;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 146))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 86))) {
              result[0] += -0.005095913;
            } else {
              result[0] += 0.00041055848;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 78))) {
              result[0] += -0.004392452;
            } else {
              result[0] += 0.0040476876;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 44))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
              result[0] += -0.00044016525;
            } else {
              result[0] += 0.0027606068;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 180))) {
              result[0] += -0.0027404772;
            } else {
              result[0] += -0.012484104;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 2))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += 0.1066999;
            } else {
              result[0] += 0.005611764;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
              result[0] += -0.0044804937;
            } else {
              result[0] += -0.00094216457;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 86))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 20))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
              result[0] += -0.016515603;
            } else {
              result[0] += -0.0019510689;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.04879087;
            } else {
              result[0] += 0.019489786;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 174))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += 0.028882284;
            } else {
              result[0] += 0.0029704724;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
              result[0] += -0.008776897;
            } else {
              result[0] += 0.00076872477;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 22))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 246))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
              result[0] += -0.00024617615;
            } else {
              result[0] += -0.013363178;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.012836638;
            } else {
              result[0] += -0.00578899;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 74))) {
            result[0] += 0.02552943;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 74))) {
              result[0] += 0.014522381;
            } else {
              result[0] += -0.0055740518;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 50))) {
          result[0] += -0.024675723;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 72))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
              result[0] += 0.036256976;
            } else {
              result[0] += -0.010159162;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 186))) {
              result[0] += -0.0034239579;
            } else {
              result[0] += 0.0019339178;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 130))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 126))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 124))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
              result[0] += 0.0021977962;
            } else {
              result[0] += -0.0026835771;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
              result[0] += 0.014414914;
            } else {
              result[0] += 0.010211523;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += -0.008403643;
            } else {
              result[0] += -0.024761902;
            }
          } else {
            result[0] += 0.0032713003;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
            result[0] += -0.02039499;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.009458961;
            } else {
              result[0] += -0.0018666508;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
            result[0] += 0.0327911;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.036110453;
            } else {
              result[0] += 0.0033109114;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 48))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 150))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 48))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 104))) {
              result[0] += 0.010583217;
            } else {
              result[0] += -0.01392099;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
              result[0] += 0.032042317;
            } else {
              result[0] += 0.011768063;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 154))) {
            result[0] += -0.04997726;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
              result[0] += -0.008824881;
            } else {
              result[0] += 0.0048329984;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 202))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += 0.00096224155;
            } else {
              result[0] += -0.009902506;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 146))) {
              result[0] += 0.02890099;
            } else {
              result[0] += 5.074874e-05;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
            result[0] += -0.0082296785;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 96))) {
              result[0] += 0.0020531386;
            } else {
              result[0] += -0.001512293;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
        result[0] += -0.009371125;
      } else {
        result[0] += 0.016960299;
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
        result[0] += -0.008832243;
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 254))) {
              result[0] += 0.008071503;
            } else {
              result[0] += 0.013252865;
            }
          } else {
            result[0] += -0.0039770114;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
              result[0] += -0.0074452083;
            } else {
              result[0] += 0.0010280218;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
              result[0] += 0.020828962;
            } else {
              result[0] += 0.006464574;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 244))) {
        result[0] += 0.044012897;
      } else {
        result[0] += -0.01173665;
      }
    } else {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 76))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
              result[0] += 0.013455394;
            } else {
              result[0] += -0.014296198;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
              result[0] += 0.00890102;
            } else {
              result[0] += -0.00038118512;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
              result[0] += 0.0076601445;
            } else {
              result[0] += 0.037503462;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
              result[0] += 0.00028127801;
            } else {
              result[0] += -0.0077372934;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
          result[0] += -0.0058348128;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
            result[0] += -0.0035285298;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
              result[0] += 0.0027004466;
            } else {
              result[0] += 0.009092315;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 52))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 4))) {
              result[0] += 0.009767297;
            } else {
              result[0] += -0.008808373;
            }
          } else {
            result[0] += 0.009121577;
          }
        } else {
          result[0] += -0.0003377227;
        }
      } else {
        result[0] += -0.0081705395;
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 112))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 110))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.012397877;
            } else {
              result[0] += 0.003961118;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 88))) {
              result[0] += -0.003340742;
            } else {
              result[0] += 0.001318198;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
            result[0] += 0.04594936;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
              result[0] += -0.012992064;
            } else {
              result[0] += 0.013072202;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 146))) {
            result[0] += -0.016053863;
          } else {
            result[0] += -0.0013210791;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 120))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 72))) {
              result[0] += -0.0011712526;
            } else {
              result[0] += 0.010021153;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 124))) {
              result[0] += -0.007040714;
            } else {
              result[0] += 0.0005415445;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 8))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
            result[0] += 0.0068953778;
          } else {
            result[0] += -0.0090838885;
          }
        } else {
          result[0] += 0.024001935;
        }
      } else {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
              result[0] += -0.009051267;
            } else {
              result[0] += -0.02782002;
            }
          } else {
            result[0] += 0.007848546;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 8))) {
              result[0] += 0.0028921093;
            } else {
              result[0] += 0.023708088;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
              result[0] += 0.0013097594;
            } else {
              result[0] += -0.001009873;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 190))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
            result[0] += 0.017365938;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += -0.026752142;
            } else {
              result[0] += -0.00315145;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
              result[0] += -0.0013150068;
            } else {
              result[0] += 0.05037552;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
              result[0] += -0.02626193;
            } else {
              result[0] += 0.030513177;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 8))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 44))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.0053047254;
            } else {
              result[0] += -0.027583335;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 54))) {
              result[0] += 0.043980025;
            } else {
              result[0] += -0.006328375;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
              result[0] += 0.0048042852;
            } else {
              result[0] += 0.042036824;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.012095607;
            } else {
              result[0] += -0.00045656387;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 236))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 190))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 112))) {
              result[0] += -0.0005225084;
            } else {
              result[0] += 0.0013189358;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
              result[0] += -0.01325873;
            } else {
              result[0] += 0.008976769;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 98))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 72))) {
              result[0] += -0.004084352;
            } else {
              result[0] += 0.0038890864;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 204))) {
              result[0] += -0.010790448;
            } else {
              result[0] += -0.0041690473;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 214))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
              result[0] += 0.00092866377;
            } else {
              result[0] += -0.021240106;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
              result[0] += 0.016957764;
            } else {
              result[0] += 0.009066544;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 56))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 120))) {
              result[0] += -0.008545898;
            } else {
              result[0] += -0.0013856884;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 216))) {
              result[0] += -0.0058118394;
            } else {
              result[0] += 0.0030270133;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.004897514;
            } else {
              result[0] += 0.012878601;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 136))) {
              result[0] += -0.009626532;
            } else {
              result[0] += -0.005689178;
            }
          }
        } else {
          result[0] += -0.04376392;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 136))) {
          result[0] += 0.03378177;
        } else {
          result[0] += 0.01186064;
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 120))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
              result[0] += -0.0032555745;
            } else {
              result[0] += 0.0158323;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 188))) {
              result[0] += 0.00046772003;
            } else {
              result[0] += 0.0077147195;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
            result[0] += 0.0011735325;
          } else {
            result[0] += -0.006349434;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 146))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.023203839;
            } else {
              result[0] += 0.00827275;
            }
          } else {
            result[0] += 0.039015923;
          }
        } else {
          result[0] += 0.016087564;
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 156))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 206))) {
          result[0] += -0.0005247008;
        } else {
          result[0] += -0.0068671047;
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 188))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 182))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
              result[0] += -0.00041580052;
            } else {
              result[0] += 0.005611544;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
              result[0] += -0.015564476;
            } else {
              result[0] += -0.005183963;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 172))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 218))) {
              result[0] += 0.023443257;
            } else {
              result[0] += -0.002142666;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
              result[0] += 0.05721947;
            } else {
              result[0] += 0.000846235;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 8))) {
              result[0] += -0.010377513;
            } else {
              result[0] += 0.002864465;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 180))) {
              result[0] += -0.041931864;
            } else {
              result[0] += -0.01965585;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
            result[0] += -0.0147984475;
          } else {
            result[0] += 0.031754512;
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 2))) {
            result[0] += 0.0027872298;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
              result[0] += -0.0018927314;
            } else {
              result[0] += 0.00035012572;
            }
          }
        } else {
          result[0] += 0.011900772;
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
        result[0] += 0.02755776;
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 22))) {
            result[0] += -0.03298561;
          } else {
            result[0] += -0.008792006;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 260))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 166))) {
              result[0] += -0.00033489856;
            } else {
              result[0] += 0.006294869;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 216))) {
              result[0] += -0.00025057443;
            } else {
              result[0] += 0.005475367;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 22))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 194))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 172))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.034542117;
            } else {
              result[0] += 0.0051070247;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 38))) {
              result[0] += -0.011878603;
            } else {
              result[0] += 0.015853336;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 12))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 204))) {
              result[0] += -0.017843468;
            } else {
              result[0] += 0.0025024624;
            }
          } else {
            result[0] += -0.038026553;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 0))) {
              result[0] += 0.0004767517;
            } else {
              result[0] += -0.005413917;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
              result[0] += -0.00014004558;
            } else {
              result[0] += 0.0038309437;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
              result[0] += -0.025853286;
            } else {
              result[0] += 0.016463885;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
              result[0] += 0.008300542;
            } else {
              result[0] += -0.0022710604;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 26))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
          result[0] += 0.06824603;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            result[0] += 0.0344267;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 24))) {
              result[0] += -0.02413011;
            } else {
              result[0] += 0.009757657;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 240))) {
            result[0] += 0.04187591;
          } else {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += 0.021772258;
            } else {
              result[0] += 0.00023637361;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += -0.00018605802;
            } else {
              result[0] += 0.00892628;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
              result[0] += -0.0029304621;
            } else {
              result[0] += 0.005322406;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
            result[0] += 0.058786936;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 204))) {
              result[0] += -0.001987611;
            } else {
              result[0] += 0.0012095956;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 12))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += -0.013887591;
            } else {
              result[0] += -0.005378675;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
              result[0] += 0.004469014;
            } else {
              result[0] += -0.0019160692;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
          result[0] += -0.052244443;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 20))) {
            result[0] += 0.034024667;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 184))) {
              result[0] += -0.028885717;
            } else {
              result[0] += -0.016080601;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 28))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 196))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 206))) {
            result[0] += 0.004683802;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
              result[0] += 0.0038840864;
            } else {
              result[0] += -0.0010689719;
            }
          }
        } else {
          result[0] += 0.040237945;
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 174))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
            result[0] += 0.009488398;
          } else {
            result[0] += 0.0063879974;
          }
        } else {
          result[0] += 0.040055256;
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 240))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 124))) {
              result[0] += 0.0063549983;
            } else {
              result[0] += 0.0503232;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 22))) {
              result[0] += -0.0018827593;
            } else {
              result[0] += 0.0051954878;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 232))) {
            result[0] += 0.032296978;
          } else {
            result[0] += 0.0139222685;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
            result[0] += 0.04671686;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
              result[0] += 0.0009886044;
            } else {
              result[0] += 0.006239516;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 30))) {
            result[0] += 0.031537697;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
              result[0] += -0.019081848;
            } else {
              result[0] += -0.0026591418;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += 0.026619418;
            } else {
              result[0] += 0.0041615735;
            }
          } else {
            result[0] += 0.047726057;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
              result[0] += 0.001571185;
            } else {
              result[0] += -0.00050108694;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 140))) {
              result[0] += 0.009636164;
            } else {
              result[0] += 0.0010100096;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += -0.0026403468;
            } else {
              result[0] += 0.0114326365;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 50))) {
              result[0] += -0.021934291;
            } else {
              result[0] += -0.0050015096;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 238))) {
              result[0] += -0.007851736;
            } else {
              result[0] += 0.03168834;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
              result[0] += 0.033044577;
            } else {
              result[0] += 9.3673785e-05;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 240))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 188))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 156))) {
              result[0] += 0.00030188911;
            } else {
              result[0] += 0.002446159;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
              result[0] += -0.0055051325;
            } else {
              result[0] += 0.0012621379;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 114))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += 0.014215629;
            } else {
              result[0] += 0.005867033;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 86))) {
              result[0] += -0.006319243;
            } else {
              result[0] += 0.0017076422;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 166))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 218))) {
            result[0] += -0.0017003525;
          } else {
            result[0] += 0.010260985;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 236))) {
            result[0] += -0.015775895;
          } else {
            result[0] += -0.007966612;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
          result[0] += 0.0038435606;
        } else {
          result[0] += 0.0005556598;
        }
      } else {
        result[0] += 0.0073271217;
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 230))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 196))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
              result[0] += -0.00010884991;
            } else {
              result[0] += -0.0013555246;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
              result[0] += 0.00045378748;
            } else {
              result[0] += 0.00696068;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 180))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
              result[0] += 0.0034289937;
            } else {
              result[0] += -0.005728068;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
              result[0] += -0.009315341;
            } else {
              result[0] += -0.004357322;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
            result[0] += -0.024190784;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
              result[0] += -0;
            } else {
              result[0] += -0.019713525;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 220))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 64))) {
              result[0] += -0.0078073195;
            } else {
              result[0] += 5.9448754e-05;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 186))) {
              result[0] += -0.006281379;
            } else {
              result[0] += -0.010314364;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            result[0] += 0.001542045;
          } else {
            result[0] += -0.0030066923;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
            result[0] += 0.045451414;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 250))) {
              result[0] += -0.0048594624;
            } else {
              result[0] += 0.002339422;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 240))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 236))) {
              result[0] += 0.0038202107;
            } else {
              result[0] += -0.0043049804;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 222))) {
              result[0] += 0.035619907;
            } else {
              result[0] += 0.011654694;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
              result[0] += 0.011329205;
            } else {
              result[0] += -0.0026530891;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
              result[0] += 0.00091164187;
            } else {
              result[0] += 0.0065599764;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 180))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
              result[0] += 0.0003863835;
            } else {
              result[0] += 0.012511316;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 178))) {
              result[0] += -0.012533942;
            } else {
              result[0] += 0.000940961;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
            result[0] += 0.02408296;
          } else {
            result[0] += 0.006255468;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 218))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 188))) {
              result[0] += 0.01437206;
            } else {
              result[0] += 0.005517607;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 156))) {
              result[0] += -0.004864832;
            } else {
              result[0] += 0.004785055;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 206))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
              result[0] += -0.008446374;
            } else {
              result[0] += 3.6487032e-05;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 86))) {
              result[0] += -0.013399954;
            } else {
              result[0] += 0.0013180381;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 228))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 56))) {
              result[0] += -0.01676119;
            } else {
              result[0] += 0.019005135;
            }
          } else {
            result[0] += 0.026536051;
          }
        } else {
          result[0] += -0.017303595;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 128))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 234))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 76))) {
              result[0] += -0.005325236;
            } else {
              result[0] += 0.0026667507;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
              result[0] += 0.006688422;
            } else {
              result[0] += 0.0130848605;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 200))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 180))) {
              result[0] += -6.232897e-05;
            } else {
              result[0] += -0.011703821;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 232))) {
              result[0] += 0.01631069;
            } else {
              result[0] += 0.0024924437;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 118))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.020456076;
            } else {
              result[0] += 1.7096074e-05;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += -0.0043996517;
            } else {
              result[0] += 0.037505116;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
              result[0] += 0.03947706;
            } else {
              result[0] += 0.00013893798;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
              result[0] += -0.00020466889;
            } else {
              result[0] += -0.0056823995;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 90))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 94))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 94))) {
              result[0] += 0.0015122422;
            } else {
              result[0] += 0.014326178;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 98))) {
              result[0] += -0.03659178;
            } else {
              result[0] += -0.00897372;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 234))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 170))) {
              result[0] += -0.0012049631;
            } else {
              result[0] += -0.010048345;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
              result[0] += 0.023757316;
            } else {
              result[0] += -0.00096084067;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
              result[0] += 0.007543098;
            } else {
              result[0] += 0.05037853;
            }
          } else {
            result[0] += -0.020506142;
          }
        } else {
          result[0] += 0.041784402;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 106))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 168))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
              result[0] += 0.022379946;
            } else {
              result[0] += 0.001123256;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 176))) {
              result[0] += -0.0040573548;
            } else {
              result[0] += -0.00018302095;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += 0.026662258;
            } else {
              result[0] += 0.001635097;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 172))) {
              result[0] += 0.0157876;
            } else {
              result[0] += 0.00078219484;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 168))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
              result[0] += -0.003070421;
            } else {
              result[0] += 0.003986802;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += -0.016073924;
            } else {
              result[0] += -0.00033199994;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 82))) {
            result[0] += -0.0026495587;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 180))) {
              result[0] += 0.01088603;
            } else {
              result[0] += 0.0028157744;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 128))) {
          result[0] += 0.035645917;
        } else {
          result[0] += 0.011242066;
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 158))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 120))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += -0.004172708;
            } else {
              result[0] += -0.013315861;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += 0.0016847762;
            } else {
              result[0] += -0.019520441;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 116))) {
            result[0] += -0.0012042717;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += 0.010144642;
            } else {
              result[0] += 0.021312661;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 122))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 162))) {
              result[0] += -0.0060045626;
            } else {
              result[0] += -0.019477958;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 154))) {
              result[0] += -0.010557742;
            } else {
              result[0] += -0.019895343;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 164))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 162))) {
              result[0] += 0.0023980096;
            } else {
              result[0] += -0.0058308984;
            }
          } else {
            result[0] += 0.0054435167;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 180))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 170))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 144))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 52))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
              result[0] += 0.01054426;
            } else {
              result[0] += -0.013723351;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 190))) {
              result[0] += 0.0035272748;
            } else {
              result[0] += -0.0014124735;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 128))) {
            result[0] += -0.020531978;
          } else {
            result[0] += -0.008468418;
          }
        }
      } else {
        result[0] += 0.027421832;
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 108))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 136))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
              result[0] += -0.0014708912;
            } else {
              result[0] += 0.010910965;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
              result[0] += 0.02383196;
            } else {
              result[0] += -0.005580553;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 192))) {
              result[0] += -0.015012925;
            } else {
              result[0] += -0.038420197;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 102))) {
              result[0] += 0.027166018;
            } else {
              result[0] += -0.0051050577;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 160))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 158))) {
              result[0] += 0.005120454;
            } else {
              result[0] += -0.0066694873;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
              result[0] += 0.040612247;
            } else {
              result[0] += 0.010608687;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 206))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 186))) {
              result[0] += 9.48829e-05;
            } else {
              result[0] += -0.0028726817;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 222))) {
              result[0] += 0.0036865217;
            } else {
              result[0] += -0.0017621756;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 12))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
              result[0] += 0.010326921;
            } else {
              result[0] += -6.580556e-05;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 14))) {
              result[0] += 0.035937645;
            } else {
              result[0] += 0.0061092353;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += 0.09306122;
            } else {
              result[0] += 0.004483877;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 12))) {
              result[0] += -0.0021131127;
            } else {
              result[0] += -0.0002122157;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 16))) {
              result[0] += -0.0010388399;
            } else {
              result[0] += 0.04628381;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 170))) {
              result[0] += -0.012791668;
            } else {
              result[0] += 0.012744421;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 14))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
              result[0] += 0.001258762;
            } else {
              result[0] += -0.010475395;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 18))) {
              result[0] += 0.025982514;
            } else {
              result[0] += 0.0025854723;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 26))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 96))) {
              result[0] += -0.020725088;
            } else {
              result[0] += 0.014970714;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 176))) {
              result[0] += -0.0017299574;
            } else {
              result[0] += 0.026476989;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 158))) {
            result[0] += -0.004242038;
          } else {
            result[0] += -0.029603764;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 100))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 126))) {
              result[0] += 0.0020087094;
            } else {
              result[0] += -0.001871386;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 44))) {
              result[0] += 0.014256491;
            } else {
              result[0] += 0.007524089;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 94))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 206))) {
              result[0] += -0.014824388;
            } else {
              result[0] += -0.0017853121;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += -0.004817825;
            } else {
              result[0] += 0.00016148947;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.0088046715;
  }
  if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 168))) {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 154))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 146))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += 9.3958115e-05;
            } else {
              result[0] += 0.0067572976;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 222))) {
              result[0] += -0.0049092458;
            } else {
              result[0] += -0.0005594916;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 178))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 108))) {
              result[0] += 0.00977688;
            } else {
              result[0] += 0.0015517873;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 204))) {
              result[0] += -0.008679122;
            } else {
              result[0] += -0.002352368;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 158))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
              result[0] += 0.009798234;
            } else {
              result[0] += 0.0032348556;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 182))) {
              result[0] += 0.014364332;
            } else {
              result[0] += 0.025256773;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 118))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 46))) {
              result[0] += -0.019178228;
            } else {
              result[0] += -0.0023992255;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
              result[0] += 0.026579231;
            } else {
              result[0] += 0.0040463707;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 172))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 170))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
              result[0] += -0.012679273;
            } else {
              result[0] += -0.03488743;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 162))) {
              result[0] += 0.012128734;
            } else {
              result[0] += -0.0036993285;
            }
          }
        } else {
          result[0] += -0.026699487;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 164))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 46))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
              result[0] += 0.008520793;
            } else {
              result[0] += -0.0012307116;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 174))) {
              result[0] += -0.0020589442;
            } else {
              result[0] += -0.0074132094;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 22))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 258))) {
              result[0] += -0.00428136;
            } else {
              result[0] += 0.042908087;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
              result[0] += 0.034873586;
            } else {
              result[0] += 0.0018836738;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 190))) {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
          result[0] += -0.0036578246;
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 208))) {
            result[0] += 0.020423703;
          } else {
            result[0] += 0.030946225;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
            result[0] += -0.006327176;
          } else {
            result[0] += -0.015637185;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 258))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 232))) {
              result[0] += 0.0020563873;
            } else {
              result[0] += 0.009788949;
            }
          } else {
            result[0] += -0.0091839805;
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 194))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
          result[0] += 0.051798742;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 252))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.006234445;
            } else {
              result[0] += 0.008611916;
            }
          } else {
            result[0] += 0.0055716326;
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 196))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 194))) {
            result[0] += 0.003886501;
          } else {
            result[0] += 0.008659224;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 198))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
              result[0] += -0.03154314;
            } else {
              result[0] += -0.0031137222;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
              result[0] += 0.00038270376;
            } else {
              result[0] += 0.0020157003;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
      result[0] += 0.026274556;
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
            result[0] += -0.014014624;
          } else {
            result[0] += 0.0017416707;
          }
        } else {
          result[0] += -0.020999283;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
          result[0] += 0.019021433;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 2))) {
              result[0] += 0.00134224;
            } else {
              result[0] += 0.0040796655;
            }
          } else {
            result[0] += -0.0030546128;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 90))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 46))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
              result[0] += -0.0019856154;
            } else {
              result[0] += -0.0002087965;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
              result[0] += 0.013158979;
            } else {
              result[0] += -0.0030576333;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
              result[0] += 0.003060523;
            } else {
              result[0] += 0.0005930412;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 48))) {
              result[0] += 0.03627108;
            } else {
              result[0] += -0.0035451804;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
              result[0] += -0.0046488056;
            } else {
              result[0] += 0.00013075671;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 120))) {
              result[0] += 0.012213302;
            } else {
              result[0] += 0.04960065;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
              result[0] += -0.0069404887;
            } else {
              result[0] += -0.01752409;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
              result[0] += 0.0142224645;
            } else {
              result[0] += -0.0028541964;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
              result[0] += -0.00048694783;
            } else {
              result[0] += -0.01978219;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
              result[0] += 0.023798453;
            } else {
              result[0] += 0.0028695017;
            }
          }
        } else {
          result[0] += 0.049870964;
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
              result[0] += -0.011660507;
            } else {
              result[0] += 0.020355195;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
              result[0] += -0.010959279;
            } else {
              result[0] += -0.022352552;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 104))) {
              result[0] += 0.0019975018;
            } else {
              result[0] += 0.02552997;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
              result[0] += -0.005593583;
            } else {
              result[0] += 0.0001567727;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 240))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 36))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
              result[0] += 0.0025115781;
            } else {
              result[0] += 0.01276709;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
              result[0] += 0.0009829837;
            } else {
              result[0] += -0.0020405638;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 176))) {
              result[0] += -0.0027318567;
            } else {
              result[0] += 0.00036724083;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
              result[0] += 0.00952085;
            } else {
              result[0] += 0.00021925311;
            }
          }
        }
      } else {
        result[0] += 0.0028065902;
      }
    } else {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 264))) {
          result[0] += -0.010198384;
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 68))) {
            result[0] += 0.008980814;
          } else {
            result[0] += -0.00083386357;
          }
        }
      } else {
        result[0] += 0.034579825;
      }
    }
  } else {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 228))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 214))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
              result[0] += -0.0005227964;
            } else {
              result[0] += 0.00078901247;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 264))) {
              result[0] += -0.0031115639;
            } else {
              result[0] += -0.008557741;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 232))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 80))) {
              result[0] += 0.044274148;
            } else {
              result[0] += 0.004722073;
            }
          } else {
            result[0] += -0.020457005;
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 208))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 214))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
              result[0] += -0.011751657;
            } else {
              result[0] += -0.004513866;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 174))) {
              result[0] += 0.0004462137;
            } else {
              result[0] += 0.0044212313;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 202))) {
              result[0] += -0.015617072;
            } else {
              result[0] += -0.0095054805;
            }
          } else {
            result[0] += -3.9380015e-05;
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 184))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 160))) {
              result[0] += 0.002189336;
            } else {
              result[0] += 0.011331045;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 230))) {
              result[0] += 0.0016538227;
            } else {
              result[0] += -0.0024343396;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 186))) {
            result[0] += 0.030128485;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
              result[0] += 0.0012932423;
            } else {
              result[0] += 0.004705872;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
          result[0] += 0.001994251;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 252))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 138))) {
              result[0] += -0.00304999;
            } else {
              result[0] += 0.011378044;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
              result[0] += 0.004555874;
            } else {
              result[0] += -0.00066039886;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
      result[0] += 0.020459097;
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 234))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 56))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
              result[0] += -0.006080057;
            } else {
              result[0] += 0.0015269167;
            }
          } else {
            result[0] += 0.004850093;
          }
        } else {
          result[0] += 0.0372449;
        }
      } else {
        result[0] += -0.0033630708;
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += 0.0020560354;
            } else {
              result[0] += -0.01611949;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
              result[0] += 0.02037514;
            } else {
              result[0] += 0.0024216073;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
              result[0] += -1.3009435e-06;
            } else {
              result[0] += -0.010717098;
            }
          } else {
            result[0] += -0.026966467;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
          result[0] += -0.035333917;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 60))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 34))) {
              result[0] += -0.0039970325;
            } else {
              result[0] += -0.018680593;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 52))) {
              result[0] += 0.060415726;
            } else {
              result[0] += -0.0035995846;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 12))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
          result[0] += 0.079136275;
        } else {
          result[0] += 0.006364348;
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.009026817;
            } else {
              result[0] += -0.00017536264;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 258))) {
              result[0] += 0.0025470634;
            } else {
              result[0] += -0.0051941304;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 78))) {
              result[0] += -0.0008757123;
            } else {
              result[0] += -0.008570026;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 52))) {
              result[0] += 0.005017677;
            } else {
              result[0] += -7.974629e-05;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 212))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 190))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 174))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += -8.178638e-05;
            } else {
              result[0] += 0.0078121508;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
              result[0] += -0.005830926;
            } else {
              result[0] += -0.0010021594;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 262))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 244))) {
              result[0] += 0.0016010083;
            } else {
              result[0] += 0.00825313;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 206))) {
              result[0] += -0.003538803;
            } else {
              result[0] += 0.0063129514;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 198))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 190))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 170))) {
              result[0] += -0.005545571;
            } else {
              result[0] += -0.0232496;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 52))) {
              result[0] += 0.012393202;
            } else {
              result[0] += -4.4827804e-05;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 208))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 200))) {
              result[0] += 0.003246121;
            } else {
              result[0] += -0.00059273635;
            }
          } else {
            result[0] += -0.013444054;
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 70))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 180))) {
              result[0] += 0.00011026966;
            } else {
              result[0] += -0.012288004;
            }
          } else {
            result[0] += 0.009097963;
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 210))) {
            result[0] += -0.004248265;
          } else {
            result[0] += -0.008213413;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
          result[0] += 0.041847315;
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 214))) {
              result[0] += 0.008025631;
            } else {
              result[0] += 0.0023118325;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 238))) {
              result[0] += -0.004090266;
            } else {
              result[0] += 0.001676759;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.0124376435;
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 148))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 142))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 130))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 126))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
              result[0] += 8.788087e-05;
            } else {
              result[0] += -0.0007693504;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 62))) {
              result[0] += 0.028994996;
            } else {
              result[0] += 0.0019165861;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 82))) {
              result[0] += 0.0018343481;
            } else {
              result[0] += -0.001085312;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += -0.00844089;
            } else {
              result[0] += -0.0010173704;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 118))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
            result[0] += 0.03216304;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.019660346;
            } else {
              result[0] += 0.010661392;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
            result[0] += -0.011746697;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 120))) {
              result[0] += 0.040727697;
            } else {
              result[0] += 0.007775008;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 146))) {
          result[0] += -0.017095553;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
            result[0] += -0.010490341;
          } else {
            result[0] += -0.0029303418;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
          result[0] += 0.0037032042;
        } else {
          result[0] += 0.019236496;
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 88))) {
          result[0] += -0.00065161986;
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 154))) {
              result[0] += 0.013047007;
            } else {
              result[0] += 0.018410644;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
              result[0] += -0.0025795808;
            } else {
              result[0] += 0.0025427365;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 44))) {
          result[0] += 0.0063687847;
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 158))) {
            result[0] += 0.021263849;
          } else {
            result[0] += 0.037114166;
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 12))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
            result[0] += -0.007888737;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.0043152072;
            } else {
              result[0] += -0.00065306626;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 184))) {
              result[0] += 0.0002832889;
            } else {
              result[0] += 0.023496693;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 164))) {
              result[0] += -0.004027024;
            } else {
              result[0] += 0.0036689022;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 52))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 168))) {
              result[0] += -0.006764188;
            } else {
              result[0] += -0.015382841;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 188))) {
              result[0] += -0.0006603152;
            } else {
              result[0] += -0.0053936313;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 160))) {
              result[0] += 0.0016898311;
            } else {
              result[0] += 0.010389189;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
              result[0] += -0.0190918;
            } else {
              result[0] += 6.013682e-05;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 284))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 30))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 258))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 30))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0022224819;
            } else {
              result[0] += -9.1448324e-05;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 64))) {
              result[0] += 0.008647813;
            } else {
              result[0] += 0.0013861794;
            }
          }
        } else {
          result[0] += -0.0051472313;
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
          result[0] += 0.022298444;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
            result[0] += 0.014823428;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 78))) {
              result[0] += 0.0028555428;
            } else {
              result[0] += -0.0060178605;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 8))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 204))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 144))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 94))) {
              result[0] += -0.011365654;
            } else {
              result[0] += 0.008808569;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 166))) {
              result[0] += -0.024276486;
            } else {
              result[0] += -0.0131767215;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 86))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
              result[0] += 0.00949425;
            } else {
              result[0] += -0.013411551;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
              result[0] += -0;
            } else {
              result[0] += 0.012317988;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
              result[0] += 0.0853606;
            } else {
              result[0] += 0.0014590746;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 42))) {
              result[0] += -0.04024057;
            } else {
              result[0] += 0.025248704;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
              result[0] += -0.00020014495;
            } else {
              result[0] += -0.008825987;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 32))) {
              result[0] += -0.0026628354;
            } else {
              result[0] += 0.00010321446;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
      result[0] += -0.002153416;
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
        result[0] += -0.0011085857;
      } else {
        result[0] += 0.00558898;
      }
    }
  }
  if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 64))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 106))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 30))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
              result[0] += -0.0072398344;
            } else {
              result[0] += -0.00012400204;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
              result[0] += 0.025649467;
            } else {
              result[0] += -0.0001327928;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 94))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
              result[0] += -0.0015077029;
            } else {
              result[0] += -0.014433789;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.026124759;
            } else {
              result[0] += 0.0094175115;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 98))) {
              result[0] += -0.009028183;
            } else {
              result[0] += -0.05055972;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
              result[0] += 0.008288766;
            } else {
              result[0] += -0.03171199;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 74))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.029759515;
            } else {
              result[0] += -0.019024173;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 86))) {
              result[0] += 0.0045653116;
            } else {
              result[0] += 6.555731e-05;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 106))) {
        result[0] += 0.029950727;
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
            result[0] += 0.0046233633;
          } else {
            result[0] += 0.020635402;
          }
        } else {
          result[0] += 0.0056214184;
        }
      }
    }
  } else {
    if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
          result[0] += -0.03250946;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
            result[0] += 0.013577218;
          } else {
            result[0] += -0.031636387;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 62))) {
          result[0] += -0.005065638;
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
            result[0] += -0.004201682;
          } else {
            result[0] += 0.0023832265;
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 84))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 84))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 152))) {
            result[0] += -0.023720464;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
              result[0] += -0.0023072253;
            } else {
              result[0] += 0.0019047937;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 158))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 116))) {
              result[0] += -0.0026599832;
            } else {
              result[0] += 0.03371409;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 98))) {
              result[0] += -0.010501557;
            } else {
              result[0] += 0.03556975;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 92))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 96))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += 0.005952064;
            } else {
              result[0] += 0.024354918;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += 0.0024803034;
            } else {
              result[0] += -0.007694894;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 110))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 56))) {
              result[0] += 0.016107908;
            } else {
              result[0] += -0.0016387564;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
              result[0] += 0.003588716;
            } else {
              result[0] += -1.9828594e-05;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 206))) {
      result[0] += 0.0046148184;
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
        result[0] += 0.0024027901;
      } else {
        result[0] += 0.00039783234;
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
          result[0] += 0.044661757;
        } else {
          result[0] += 0.0023485457;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
              result[0] += -0.04294611;
            } else {
              result[0] += -0.02216121;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += 0.0008537049;
            } else {
              result[0] += -0.020757614;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 192))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 134))) {
              result[0] += -0.0010473432;
            } else {
              result[0] += -0.012557432;
            }
          } else {
            result[0] += 0.0345631;
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 10))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
          result[0] += 0.009390167;
        } else {
          result[0] += 0.0029766764;
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 12))) {
          result[0] += -0.00874328;
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
              result[0] += -0.011705816;
            } else {
              result[0] += 0.0072610932;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
              result[0] += -0.00043213202;
            } else {
              result[0] += 0.0003012484;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 14))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
            result[0] += -0.0078075933;
          } else {
            result[0] += -0.03296116;
          }
        } else {
          result[0] += 0.0045325574;
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 12))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
            result[0] += 0.0027738758;
          } else {
            result[0] += -0.0002652021;
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.017156854;
            } else {
              result[0] += 0.0061222934;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
              result[0] += 0.009275707;
            } else {
              result[0] += -0.008091964;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
            result[0] += 0.02927331;
          } else {
            result[0] += 0.0036204413;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
              result[0] += -0.009472274;
            } else {
              result[0] += -0.036276244;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 76))) {
              result[0] += -0.0055947257;
            } else {
              result[0] += -0.016915802;
            }
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += 0.0031531143;
            } else {
              result[0] += -0.020101098;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 10))) {
              result[0] += -0.019649826;
            } else {
              result[0] += -0.0027359962;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
              result[0] += 0.02526692;
            } else {
              result[0] += -0;
            }
          } else {
            result[0] += 0.0012190904;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 12))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
        result[0] += 0.079478934;
      } else {
        result[0] += 0.004041801;
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.05100611;
            } else {
              result[0] += 0.013841884;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
              result[0] += -0.0001244545;
            } else {
              result[0] += 0.0013780014;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
              result[0] += 0.004001136;
            } else {
              result[0] += -0.011721842;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
              result[0] += 0.015507097;
            } else {
              result[0] += -0.002207271;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 68))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 174))) {
              result[0] += -0.0029817559;
            } else {
              result[0] += 0.0030468712;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 154))) {
              result[0] += 0.008323856;
            } else {
              result[0] += 0.0378311;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
              result[0] += 0.019646537;
            } else {
              result[0] += -0.009434113;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 80))) {
              result[0] += 0.015092474;
            } else {
              result[0] += 0.00023066562;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 26))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 12))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 14))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
              result[0] += 0.00044216277;
            } else {
              result[0] += 0.0030401708;
            }
          } else {
            result[0] += -0.053345073;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
              result[0] += 0.024727738;
            } else {
              result[0] += -0.008567747;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 38))) {
              result[0] += 0.036883164;
            } else {
              result[0] += 0.014384936;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            result[0] += -0.004501282;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 14))) {
              result[0] += -0.023340857;
            } else {
              result[0] += 0.0061771907;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 76))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.024513956;
            } else {
              result[0] += -0.0015961519;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
              result[0] += 0.0052294037;
            } else {
              result[0] += 0.00016616772;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 218))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 62))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
            result[0] += 0.0042944904;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 122))) {
              result[0] += -0.035668056;
            } else {
              result[0] += 0.033606566;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
            result[0] += 0.013674865;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 98))) {
              result[0] += 0.0019538465;
            } else {
              result[0] += -0.001950983;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 6))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
            result[0] += 0.0020924758;
          } else {
            result[0] += 0.00730929;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 78))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 10))) {
              result[0] += 0.00075499614;
            } else {
              result[0] += 0.0019689128;
            }
          } else {
            result[0] += -0.0016815666;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 58))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 226))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 8))) {
              result[0] += -0.0115894945;
            } else {
              result[0] += -0.051073868;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
              result[0] += 0.011092906;
            } else {
              result[0] += 0.00051702786;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 210))) {
              result[0] += -0.013087346;
            } else {
              result[0] += -0.0062652132;
            }
          } else {
            result[0] += 0.009260368;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 92))) {
          result[0] += -0.013211399;
        } else {
          result[0] += -0.005936895;
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 64))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
            result[0] += 0.018809779;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
              result[0] += -0.00044671824;
            } else {
              result[0] += 0.0025783246;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
              result[0] += -0.008156588;
            } else {
              result[0] += 0.011215605;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
              result[0] += 0.009001787;
            } else {
              result[0] += 0.027359499;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
            result[0] += 0.03523052;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 52))) {
              result[0] += -0.016149698;
            } else {
              result[0] += -0.0014326718;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 82))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 172))) {
              result[0] += 0.009203473;
            } else {
              result[0] += -0.01101023;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 30))) {
              result[0] += 0.007282652;
            } else {
              result[0] += -3.0036234e-05;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 240))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 160))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
              result[0] += 0.00011574587;
            } else {
              result[0] += 0.021244911;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 202))) {
              result[0] += 0.0015066937;
            } else {
              result[0] += 0.01799863;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 116))) {
            result[0] += 0.002711843;
          } else {
            result[0] += 0.01596604;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 178))) {
              result[0] += -0.0002499475;
            } else {
              result[0] += 0.0038768928;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += 0.022616291;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
            result[0] += -0.018833179;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
              result[0] += -0.007184063;
            } else {
              result[0] += -0.00079069706;
            }
          }
        }
      }
    } else {
      result[0] += 0.0025515545;
    }
  } else {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 254))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 208))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 232))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 230))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 218))) {
              result[0] += -9.1387315e-05;
            } else {
              result[0] += -0.0014537438;
            }
          } else {
            result[0] += 0.0155031625;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 142))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 140))) {
              result[0] += -0.003084584;
            } else {
              result[0] += -0.01318902;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 160))) {
              result[0] += 0.0037594668;
            } else {
              result[0] += -0.0014401844;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 88))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
              result[0] += 0.0035882175;
            } else {
              result[0] += -0.0048747966;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 162))) {
              result[0] += 0.0041894084;
            } else {
              result[0] += 0.0011686251;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 210))) {
              result[0] += -0.0026981125;
            } else {
              result[0] += -0.009141969;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
              result[0] += 0.0005261341;
            } else {
              result[0] += -0.0018949058;
            }
          }
        }
      }
    } else {
      result[0] += 0.018716332;
    }
  }
  if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 252))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 236))) {
              result[0] += 2.3203478e-05;
            } else {
              result[0] += -0.0077802497;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
              result[0] += 0.031958427;
            } else {
              result[0] += 0.0035587456;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 128))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
              result[0] += 0.002486235;
            } else {
              result[0] += 0.011185441;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 204))) {
              result[0] += -0.0068403906;
            } else {
              result[0] += -0.001759151;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 140))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.0073994272;
            } else {
              result[0] += -0.016759716;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 108))) {
              result[0] += 0.028997958;
            } else {
              result[0] += -0.00054327224;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 232))) {
              result[0] += 0.02114836;
            } else {
              result[0] += -0.011946307;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 264))) {
              result[0] += 0.002376872;
            } else {
              result[0] += -0.0016341202;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
          result[0] += -0.03481881;
        } else {
          result[0] += -0.007797651;
        }
      } else {
        result[0] += -0.00034481628;
      }
    }
  } else {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
        result[0] += 0.0057917973;
      } else {
        result[0] += 0.0218565;
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 152))) {
        result[0] += -0.001166375;
      } else {
        result[0] += 0.0030204598;
      }
    }
  }
  if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 100))) {
              result[0] += -8.485738e-05;
            } else {
              result[0] += 0.0070774257;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 148))) {
              result[0] += -0.0021743942;
            } else {
              result[0] += -0.029421;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 172))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
              result[0] += 0.0006834134;
            } else {
              result[0] += 0.010325692;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
              result[0] += -0.001693792;
            } else {
              result[0] += 0.0011259892;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
          result[0] += -0.0077881524;
        } else {
          result[0] += 0.0026733195;
        }
      }
    } else {
      if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 166))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 32))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 72))) {
              result[0] += 0.028013662;
            } else {
              result[0] += 0.0070042475;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
              result[0] += 0.004207076;
            } else {
              result[0] += -0.016646465;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 182))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += 0.014890812;
            } else {
              result[0] += -0.0032379036;
            }
          } else {
            result[0] += 0.0058797766;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 166))) {
            result[0] += -0.004874174;
          } else {
            result[0] += -0.014910466;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 194))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 178))) {
              result[0] += -0.0024306828;
            } else {
              result[0] += -0.021217367;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 232))) {
              result[0] += 0.0033338335;
            } else {
              result[0] += -0.00023477618;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 194))) {
            result[0] += -0.037188523;
          } else {
            result[0] += -0.012556768;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 4))) {
            result[0] += 0.0015031213;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
              result[0] += 0.038557947;
            } else {
              result[0] += 0.018226711;
            }
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 84))) {
          result[0] += -0.02505053;
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 204))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 116))) {
              result[0] += -0.0005712052;
            } else {
              result[0] += -0.0054953173;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 216))) {
              result[0] += 0.0013088783;
            } else {
              result[0] += -0.0013262478;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 214))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 92))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 246))) {
            result[0] += -0.0016769763;
          } else {
            result[0] += 0.004949599;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
            result[0] += -0.022474889;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
              result[0] += 0.011304743;
            } else {
              result[0] += 0.0049235076;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 86))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 82))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
              result[0] += -0.0013205075;
            } else {
              result[0] += 0.005631945;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
              result[0] += -0.013254325;
            } else {
              result[0] += -0.0048218477;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 234))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += 0.0018386835;
            } else {
              result[0] += -0.010448969;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 246))) {
              result[0] += -0.0026523883;
            } else {
              result[0] += 0.00015089083;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 148))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 198))) {
              result[0] += 0.0024612967;
            } else {
              result[0] += -0.00267146;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 134))) {
              result[0] += 0.0072369473;
            } else {
              result[0] += 0.03716041;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 34))) {
              result[0] += 0.016798072;
            } else {
              result[0] += -0.020668425;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
              result[0] += -0.0026954925;
            } else {
              result[0] += 0.00023960511;
            }
          }
        }
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 140))) {
            if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.009252391;
            } else {
              result[0] += 0.0032986738;
            }
          } else {
            result[0] += -0.01665429;
          }
        } else {
          result[0] += 0.0028180957;
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 128))) {
        result[0] += 0.02164252;
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          result[0] += -0.012067605;
        } else {
          result[0] += 0.0026951;
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 96))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 48))) {
            result[0] += 0.015092296;
          } else {
            result[0] += -0.038058575;
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 90))) {
            result[0] += 0.00892017;
          } else {
            result[0] += 0.031467702;
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 216))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 154))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
              result[0] += 0.0022178197;
            } else {
              result[0] += -0.005610294;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
              result[0] += 0.011955344;
            } else {
              result[0] += -0.0074412758;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
            result[0] += 0.0014451804;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.0032450047;
            } else {
              result[0] += 0.00080591295;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 96))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 164))) {
          result[0] += 0.010952067;
        } else {
          result[0] += -0.003597115;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 236))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 144))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 230))) {
              result[0] += 0.0010651236;
            } else {
              result[0] += 0.019118952;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
              result[0] += -0.0010132345;
            } else {
              result[0] += 0.0005361346;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 246))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
              result[0] += 0.0063631246;
            } else {
              result[0] += -0.005013407;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 216))) {
              result[0] += -0.0018120991;
            } else {
              result[0] += 0.0013921553;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 90))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 6))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 14))) {
              result[0] += -0.0027856524;
            } else {
              result[0] += 0.011518478;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 52))) {
              result[0] += -0.024774041;
            } else {
              result[0] += -0.0019057058;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 10))) {
              result[0] += -0.01394673;
            } else {
              result[0] += 0.006964297;
            }
          } else {
            result[0] += 0.05194057;
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 152))) {
          result[0] += -0.028471593;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 186))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 20))) {
              result[0] += -0.0014699342;
            } else {
              result[0] += 0.024541676;
            }
          } else {
            result[0] += -0.010517646;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 210))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
              result[0] += -0.006757573;
            } else {
              result[0] += 0.0052805874;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 52))) {
              result[0] += -0.01105545;
            } else {
              result[0] += 0.001222587;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 72))) {
              result[0] += -0.011149263;
            } else {
              result[0] += -0.031825926;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 80))) {
              result[0] += -6.1052204e-05;
            } else {
              result[0] += -0.0018558489;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            result[0] += -0;
          } else {
            result[0] += 0.037658174;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 184))) {
            result[0] += -0.022644173;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 46))) {
              result[0] += -0.011395801;
            } else {
              result[0] += -0.004704497;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 40))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 26))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 116))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.018175911;
            } else {
              result[0] += 0.0022316524;
            }
          } else {
            result[0] += -0.01448056;
          }
        } else {
          result[0] += 0.03850323;
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 10))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
            result[0] += -0.0056385025;
          } else {
            result[0] += -0.001911032;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 142))) {
              result[0] += 0.006841138;
            } else {
              result[0] += 0.0015013752;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
              result[0] += -0.003987257;
            } else {
              result[0] += 0.0014403629;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 112))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 140))) {
            result[0] += 0.0001458845;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 168))) {
              result[0] += -0.018207422;
            } else {
              result[0] += 0.0007071493;
            }
          }
        } else {
          result[0] += -0.014048204;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
              result[0] += 0.024706252;
            } else {
              result[0] += 0.012115999;
            }
          } else {
            result[0] += -0.010074422;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 160))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
              result[0] += 0.00012816093;
            } else {
              result[0] += -0.0048219212;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
              result[0] += 0.00042065704;
            } else {
              result[0] += -0.00044052277;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 2))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
            result[0] += 0.012054322;
          } else {
            result[0] += -0.012439433;
          }
        } else {
          result[0] += 0.004142578;
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 246))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
              result[0] += 0.0005554498;
            } else {
              result[0] += 0.0034215685;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += 0.00583103;
            } else {
              result[0] += 0.00022540719;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 0))) {
              result[0] += 0.003226608;
            } else {
              result[0] += 0.029922882;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 20))) {
              result[0] += -0.006730991;
            } else {
              result[0] += -0.001442519;
            }
          }
        }
      }
    } else {
      result[0] += 0.020426724;
    }
  } else {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 254))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 16))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            result[0] += 0.0003546644;
          } else {
            result[0] += -0.0044249105;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 132))) {
              result[0] += 0.0020044944;
            } else {
              result[0] += -0.0026672701;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 10))) {
              result[0] += -0.007007061;
            } else {
              result[0] += -0.0024277617;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 156))) {
              result[0] += 0.0035193504;
            } else {
              result[0] += 0.016300239;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
              result[0] += -0.002883124;
            } else {
              result[0] += 0.0034831774;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 32))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 222))) {
              result[0] += -0.004244165;
            } else {
              result[0] += -0.02417766;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 36))) {
              result[0] += -0.0009843176;
            } else {
              result[0] += -0.00013388977;
            }
          }
        }
      }
    } else {
      result[0] += 0.013616696;
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 44))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 42))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 20))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 28))) {
              result[0] += 0.00050606584;
            } else {
              result[0] += -0.0017093392;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
              result[0] += -0.0051450143;
            } else {
              result[0] += 0.0035812582;
            }
          }
        } else {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
            result[0] += -0.010360922;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 6))) {
              result[0] += 0.0013417737;
            } else {
              result[0] += -0.0003085478;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 102))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
            result[0] += 0.06123736;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
              result[0] += -0.010781719;
            } else {
              result[0] += -0.002857905;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 52))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
              result[0] += 0.013251043;
            } else {
              result[0] += -0.020714538;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 76))) {
              result[0] += -0.0037599222;
            } else {
              result[0] += -0.0004986952;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 24))) {
        result[0] += 0.08917741;
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 38))) {
          result[0] += -0.030496597;
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 252))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 274))) {
              result[0] += -0.0025217582;
            } else {
              result[0] += 0.00018528964;
            }
          } else {
            result[0] += -0.014542065;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 222))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
            result[0] += -0.009864878;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 50))) {
              result[0] += 0.017176276;
            } else {
              result[0] += 0.030590722;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 92))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
              result[0] += 0.0058373446;
            } else {
              result[0] += -0.012375738;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.047506116;
            } else {
              result[0] += 0.011469667;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
          result[0] += 0.004630673;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 38))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 252))) {
              result[0] += -0.004851036;
            } else {
              result[0] += -0.023372188;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 50))) {
              result[0] += 0.036052324;
            } else {
              result[0] += -0.002185328;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 60))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 56))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 224))) {
              result[0] += -0.00040355555;
            } else {
              result[0] += -0.0124766445;
            }
          } else {
            result[0] += 0.008253347;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 92))) {
              result[0] += -0.011534996;
            } else {
              result[0] += -0.0021382947;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
              result[0] += -0.0014395117;
            } else {
              result[0] += 0.019073278;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
              result[0] += 0.019316873;
            } else {
              result[0] += 0.0020871195;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
              result[0] += 0.004829779;
            } else {
              result[0] += 0.021349184;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
              result[0] += -0.0045497594;
            } else {
              result[0] += 0.015120694;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 66))) {
              result[0] += 0.0015102568;
            } else {
              result[0] += -3.608707e-06;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
    if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 54))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
              result[0] += 0.00032929177;
            } else {
              result[0] += 0.029328126;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 216))) {
              result[0] += -0.00073194504;
            } else {
              result[0] += -0.009144339;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.032370485;
            } else {
              result[0] += 0.0061941766;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 50))) {
              result[0] += 0.053673845;
            } else {
              result[0] += 0.018453414;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 18))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 178))) {
              result[0] += -0.010862867;
            } else {
              result[0] += 0.0031142721;
            }
          } else {
            result[0] += 0.04961296;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 28))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 150))) {
              result[0] += -0.035367988;
            } else {
              result[0] += -0.014287199;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 18))) {
              result[0] += 0.0022301266;
            } else {
              result[0] += -0.0076341094;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 122))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 74))) {
          result[0] += 0.009890891;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 32))) {
            result[0] += -0.034998115;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 76))) {
              result[0] += 0.003447539;
            } else {
              result[0] += 0.0015428506;
            }
          }
        }
      } else {
        result[0] += 0.021317383;
      }
    }
  } else {
    if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
        result[0] += 0.06219096;
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 106))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 64))) {
              result[0] += 0.0083704125;
            } else {
              result[0] += 0.0011106033;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.006979663;
            } else {
              result[0] += -0.004151942;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
              result[0] += 0.017166223;
            } else {
              result[0] += 0.0019738798;
            }
          } else {
            result[0] += -0.025185201;
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 70))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 114))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 66))) {
              result[0] += 0.0008634852;
            } else {
              result[0] += -0.008104631;
            }
          } else {
            result[0] += 0.0040236074;
          }
        } else {
          result[0] += 0.010268874;
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 82))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 74))) {
              result[0] += -0.029302115;
            } else {
              result[0] += 0.027285008;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
              result[0] += -0.004545824;
            } else {
              result[0] += -0.00053453137;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
              result[0] += 0.0019230042;
            } else {
              result[0] += -4.2119464e-06;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
              result[0] += -0.010225962;
            } else {
              result[0] += -0.00024979832;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 34))) {
    if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 36))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 26))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 200))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 4))) {
            result[0] += -0.0014482096;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 196))) {
              result[0] += 0.0071594575;
            } else {
              result[0] += 0.027249644;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
              result[0] += 0.011023153;
            } else {
              result[0] += -0.010220974;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
              result[0] += 0.0010131482;
            } else {
              result[0] += -0.00073193485;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 208))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
            result[0] += -0.049486402;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
              result[0] += -0.00315987;
            } else {
              result[0] += -0.014758741;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 26))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 22))) {
              result[0] += 0.004376946;
            } else {
              result[0] += 0.040119316;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 16))) {
              result[0] += -0.010627906;
            } else {
              result[0] += 0.0038438165;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 38))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 16))) {
            result[0] += 0.0069011683;
          } else {
            result[0] += 0.016726343;
          }
        } else {
          result[0] += 0.0049917004;
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 258))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 186))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
              result[0] += 0.01811142;
            } else {
              result[0] += 0.0030194859;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.002317766;
            } else {
              result[0] += 0.00087052287;
            }
          }
        } else {
          result[0] += -0.0039971955;
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 40))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 162))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 38))) {
          result[0] += -0.0014173501;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 154))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 152))) {
              result[0] += -0.0050792135;
            } else {
              result[0] += 0.006009587;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 40))) {
              result[0] += -0.00039048417;
            } else {
              result[0] += -0.009521704;
            }
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 182))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 182))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 38))) {
              result[0] += -0.0080691455;
            } else {
              result[0] += 0.016010879;
            }
          } else {
            result[0] += -0.01479948;
          }
        } else {
          result[0] += -0.02853345;
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 46))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 82))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 38))) {
              result[0] += -0.015648594;
            } else {
              result[0] += 0.005108214;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
              result[0] += 0.011618692;
            } else {
              result[0] += 0.0042830543;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 200))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 42))) {
              result[0] += 0.005279971;
            } else {
              result[0] += -0.009966015;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
              result[0] += -0.0035748226;
            } else {
              result[0] += 0.0021337403;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 172))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 180))) {
              result[0] += -0.025904989;
            } else {
              result[0] += 0.0019193542;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
              result[0] += -0.00022157452;
            } else {
              result[0] += -0.0015951125;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 28))) {
              result[0] += -0.00094731955;
            } else {
              result[0] += 0.029716676;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += 0.00062934245;
            } else {
              result[0] += -0.00041988047;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 112))) {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 110))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 112))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.0063841576;
            } else {
              result[0] += 0.0017673693;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
              result[0] += -0.0013868456;
            } else {
              result[0] += 0.00015485496;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 76))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 72))) {
              result[0] += -0.009897975;
            } else {
              result[0] += -0.030677944;
            }
          } else {
            result[0] += 0.021359773;
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
          result[0] += 0.021139013;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 148))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.0034971682;
            } else {
              result[0] += -0.0043157833;
            }
          } else {
            result[0] += -0.010502246;
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 144))) {
        result[0] += 0.02943166;
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
            result[0] += 0.005999833;
          } else {
            result[0] += -0.0019856528;
          }
        } else {
          result[0] += 0.013738169;
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 70))) {
          result[0] += 0.008470384;
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
            result[0] += -0.003441482;
          } else {
            result[0] += -0.011388111;
          }
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 110))) {
          result[0] += 0.0024586066;
        } else {
          result[0] += -0.021055488;
        }
      }
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 50))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 76))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 26))) {
              result[0] += -0.026231218;
            } else {
              result[0] += -0.0044914987;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
              result[0] += 0.027324816;
            } else {
              result[0] += 0.0072242687;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 186))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 170))) {
              result[0] += -0.0015850182;
            } else {
              result[0] += -0.010035633;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 134))) {
              result[0] += 0.054534066;
            } else {
              result[0] += 0.0008897401;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 166))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
            result[0] += 0.075174846;
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
              result[0] += -0.0014329397;
            } else {
              result[0] += 0.002409567;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 180))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
              result[0] += -0.007922634;
            } else {
              result[0] += -0.00042070524;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 158))) {
              result[0] += -0.00044648108;
            } else {
              result[0] += 0.0013399239;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 180))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 168))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.0013971846;
            } else {
              result[0] += -6.5496147e-06;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
              result[0] += -0.02558857;
            } else {
              result[0] += -0.0035198168;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 68))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
              result[0] += -0.0007052002;
            } else {
              result[0] += 0.011330684;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 110))) {
              result[0] += 0.008754961;
            } else {
              result[0] += 0.017457347;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 148))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 6))) {
              result[0] += 0.021995483;
            } else {
              result[0] += 0.0037829764;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.013063647;
            } else {
              result[0] += -0.008626144;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 218))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 210))) {
              result[0] += -0.00048324605;
            } else {
              result[0] += 0.014737451;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 168))) {
              result[0] += 0.03996015;
            } else {
              result[0] += -0.004091793;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 170))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 144))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += 0.0020200752;
            } else {
              result[0] += 0.016987968;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 192))) {
              result[0] += -0.013594501;
            } else {
              result[0] += 0.0012423365;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
            result[0] += -0.0187645;
          } else {
            result[0] += -0.007347082;
          }
        }
      } else {
        result[0] += 0.021339634;
      }
    }
  } else {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 198))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 112))) {
          result[0] += 0.06586228;
        } else {
          result[0] += 0.021907127;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 94))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 142))) {
              result[0] += 0.0030923286;
            } else {
              result[0] += 0.00036826293;
            }
          } else {
            result[0] += 0.04356821;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += 0.01220308;
            } else {
              result[0] += -0.007081172;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 54))) {
              result[0] += 3.0494935e-05;
            } else {
              result[0] += -0.006945651;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 16))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 10))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
            result[0] += 0.00072253105;
          } else {
            result[0] += -0.020929243;
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 194))) {
            result[0] += 0.057745833;
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 32))) {
              result[0] += -0.0037926536;
            } else {
              result[0] += 0.042418174;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.034033656;
            } else {
              result[0] += -0.021374384;
            }
          } else {
            result[0] += -0;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 40))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += -0.0036783994;
            } else {
              result[0] += -0.012444709;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 156))) {
              result[0] += -0.00028157423;
            } else {
              result[0] += -0.0030722509;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 6))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 248))) {
        if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
          result[0] += -0.018552605;
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 212))) {
            result[0] += 0.013609043;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
              result[0] += -0.0035600197;
            } else {
              result[0] += 0.005957694;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 254))) {
          result[0] += -0.0039241104;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 42))) {
              result[0] += -0.0053416244;
            } else {
              result[0] += 0.0076100454;
            }
          } else {
            result[0] += 0.0010679305;
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 272))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 266))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 8))) {
            result[0] += -0.005295289;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 234))) {
              result[0] += -1.6435115e-05;
            } else {
              result[0] += -0.0010428902;
            }
          }
        } else {
          result[0] += -0.0050287596;
        }
      } else {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 228))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 274))) {
            result[0] += 0.025240282;
          } else {
            result[0] += 0.006482261;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 264))) {
            result[0] += 0.0052149496;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 202))) {
              result[0] += 0.006406661;
            } else {
              result[0] += -0.0011082895;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.010802027;
  }
  if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 168))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 140))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 136))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 132))) {
              result[0] += 2.1751774e-05;
            } else {
              result[0] += -0.0076580383;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 68))) {
              result[0] += -0.00075408455;
            } else {
              result[0] += 0.015074461;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += -0.0018395454;
            } else {
              result[0] += 0.015883565;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += -0.0091815125;
            } else {
              result[0] += 0.0014608675;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 172))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += 0.0026474108;
            } else {
              result[0] += 0.01692606;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 140))) {
              result[0] += 0.02202346;
            } else {
              result[0] += -0.012683655;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 228))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
              result[0] += -0.0071168602;
            } else {
              result[0] += -0.00039252793;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
              result[0] += 0.001928371;
            } else {
              result[0] += -0.0002336656;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 148))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 30))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 26))) {
              result[0] += -0.025342045;
            } else {
              result[0] += 0.020604817;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
              result[0] += -0.013844515;
            } else {
              result[0] += -0.0027904052;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 142))) {
            result[0] += 0.019259404;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 170))) {
              result[0] += 0.0053436733;
            } else {
              result[0] += 0.0006968994;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 72))) {
          result[0] += -0.013023679;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 96))) {
            result[0] += -0.0089388685;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 140))) {
              result[0] += -0.013734943;
            } else {
              result[0] += 0.0016584283;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 158))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 4))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
            result[0] += -0.00055690337;
          } else {
            result[0] += -0.016434593;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
            result[0] += 0.076187894;
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 144))) {
              result[0] += 0.0040504658;
            } else {
              result[0] += 0.010756901;
            }
          }
        }
      } else {
        result[0] += 0.023556888;
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 184))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 0))) {
              result[0] += 0.06953845;
            } else {
              result[0] += -0.0023770647;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 186))) {
              result[0] += 0.021977352;
            } else {
              result[0] += 0.0001357905;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 44))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 72))) {
              result[0] += 0.0026280086;
            } else {
              result[0] += 0.015929218;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 44))) {
              result[0] += 0.0037350792;
            } else {
              result[0] += 0.00039957697;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 18))) {
          result[0] += 0.03222883;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 176))) {
            result[0] += -0.011439302;
          } else {
            result[0] += -0.0039858185;
          }
        }
      }
    }
  }
  if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 0))) {
    if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 16))) {
      if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
        result[0] += -0.0012402734;
      } else {
        result[0] += -0.019989545;
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 8))) {
        result[0] += 0.0026303197;
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
            result[0] += 0.040178392;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.0065984237;
            } else {
              result[0] += 0.02927506;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.020329256;
            } else {
              result[0] += -0;
            }
          } else {
            result[0] += 0.012915528;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 2))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
            result[0] += 0.009343422;
          } else {
            result[0] += -0.012758999;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
            result[0] += -0.027498607;
          } else {
            result[0] += -0.012787706;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 134))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 24))) {
            result[0] += 0.013563978;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += -0;
            } else {
              result[0] += 0.017173748;
            }
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 88))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 42))) {
              result[0] += -0;
            } else {
              result[0] += -0.018669667;
            }
          } else {
            result[0] += -0;
          }
        }
      }
    } else {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 6))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
            result[0] += 0.0013438625;
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 94))) {
              result[0] += 0.0075284163;
            } else {
              result[0] += 0.020848578;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 10))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 224))) {
              result[0] += 0.0053101867;
            } else {
              result[0] += -0.005638622;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 226))) {
              result[0] += 0.00015324964;
            } else {
              result[0] += 0.0012270033;
            }
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 240))) {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 238))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 208))) {
              result[0] += -0.0004296944;
            } else {
              result[0] += 0.00052584225;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 174))) {
              result[0] += 0.0010627261;
            } else {
              result[0] += -0.00823095;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 242))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 256))) {
              result[0] += 0.0063353493;
            } else {
              result[0] += 0.00052647333;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 190))) {
              result[0] += 0.0017266994;
            } else {
              result[0] += -0.0006555208;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 94))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 80))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 34))) {
              result[0] += 0.0004357934;
            } else {
              result[0] += -0.0016803583;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
              result[0] += 0.013963197;
            } else {
              result[0] += 0.0010920282;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 70))) {
              result[0] += -0.0020354139;
            } else {
              result[0] += -0.0084240325;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 76))) {
              result[0] += 0.00018836466;
            } else {
              result[0] += -0.001789179;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 98))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 112))) {
            result[0] += -0.0021141667;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 142))) {
              result[0] += 0.0008162449;
            } else {
              result[0] += -0.0007128279;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 110))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 118))) {
              result[0] += 0.005020843;
            } else {
              result[0] += 0.015004709;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
              result[0] += 0.004781536;
            } else {
              result[0] += 0.0005883411;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 96))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 130))) {
            result[0] += -0.035336528;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
              result[0] += 0.023583809;
            } else {
              result[0] += 0.0007455337;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 90))) {
            result[0] += 0.0023794405;
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
              result[0] += 0.023832815;
            } else {
              result[0] += 0.010761608;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 116))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 146))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 196))) {
              result[0] += -0.010108376;
            } else {
              result[0] += -0.0007176217;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 106))) {
              result[0] += 0.0009595863;
            } else {
              result[0] += -0.004075228;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 148))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 144))) {
              result[0] += -0;
            } else {
              result[0] += 0.01168954;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 142))) {
              result[0] += -0.0033294098;
            } else {
              result[0] += -6.6527915e-05;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 108))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 118))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 48))) {
              result[0] += 0.033734445;
            } else {
              result[0] += 0.0013792046;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
              result[0] += 0.011527828;
            } else {
              result[0] += 0.0049809148;
            }
          }
        } else {
          result[0] += 0.013943759;
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 178))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += -0.008578103;
            } else {
              result[0] += 0.0029740678;
            }
          } else {
            result[0] += -0.018524783;
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 140))) {
              result[0] += -0.00042957798;
            } else {
              result[0] += -0.009110951;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 132))) {
              result[0] += 0.021989485;
            } else {
              result[0] += 0.0028953014;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 200))) {
        result[0] += -0.0064930986;
      } else {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 188))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 184))) {
              result[0] += 0.00023702074;
            } else {
              result[0] += 0.0032349098;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 232))) {
              result[0] += -0.0035499786;
            } else {
              result[0] += -0;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 216))) {
              result[0] += -0.008582824;
            } else {
              result[0] += 0.0003269389;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 56))) {
              result[0] += 0.006198867;
            } else {
              result[0] += 0.0029896083;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 198))) {
    if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 194))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 148))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 76))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 72))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 24))) {
              result[0] += -0.008292492;
            } else {
              result[0] += 0.048605617;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 22))) {
              result[0] += -0.011937882;
            } else {
              result[0] += -0.0048466916;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 138))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 36))) {
              result[0] += -0.0032074524;
            } else {
              result[0] += 6.964723e-05;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 52))) {
              result[0] += 0.024765873;
            } else {
              result[0] += -0.0021545556;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 110))) {
              result[0] += -1.2665853e-05;
            } else {
              result[0] += -0.023850983;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 154))) {
              result[0] += 0.0025796972;
            } else {
              result[0] += 0.00040402517;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 160))) {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += -0.0170327;
            } else {
              result[0] += -0.00829741;
            }
          } else {
            result[0] += -0.0026483634;
          }
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 86))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 188))) {
          result[0] += 0.0010909947;
        } else {
          result[0] += 0.0053282953;
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 110))) {
          result[0] += 0.026514;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 100))) {
            result[0] += 0.037232116;
          } else {
            result[0] += 0.004267938;
          }
        }
      }
    }
  } else {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 78))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 248))) {
        if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 126))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 202))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 18))) {
              result[0] += 0.026404385;
            } else {
              result[0] += -0.001118703;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
              result[0] += 0.0017804791;
            } else {
              result[0] += -0.007409166;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 130))) {
            result[0] += -0.004484429;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 222))) {
              result[0] += 0.031472586;
            } else {
              result[0] += 0.0001007054;
            }
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 4))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 252))) {
              result[0] += -0.003407167;
            } else {
              result[0] += 0.001687183;
            }
          } else {
            result[0] += 0.014353302;
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 28))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 28))) {
              result[0] += -0.0015060778;
            } else {
              result[0] += -0.0003457068;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 10))) {
              result[0] += -0.013887319;
            } else {
              result[0] += -0.0035536517;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 238))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 160))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 74))) {
              result[0] += -0.0055618854;
            } else {
              result[0] += -0.0014347414;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 224))) {
              result[0] += -0.00070218346;
            } else {
              result[0] += 0.0014398324;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 8))) {
            result[0] += 0.023142483;
          } else {
            result[0] += -0.000307219;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 24))) {
          result[0] += 0.0010922215;
        } else {
          result[0] += 0.017112693;
        }
      }
    }
  }
  if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 214))) {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 12))) {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 2))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
            result[0] += -0.010769006;
          } else {
            result[0] += 0.0069697066;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 24))) {
            result[0] += 0.02430601;
          } else {
            result[0] += 0.00639099;
          }
        }
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 62))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 20))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 34))) {
              result[0] += 0.015791424;
            } else {
              result[0] += -0.0046518813;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 36))) {
              result[0] += -0.03668975;
            } else {
              result[0] += -0.012895368;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 142))) {
              result[0] += 0.03604609;
            } else {
              result[0] += 0.012836625;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 50))) {
              result[0] += -0.010758377;
            } else {
              result[0] += -0.001967916;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 220))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 232))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 80))) {
              result[0] += 0.034449853;
            } else {
              result[0] += 0.0030996487;
            }
          } else {
            result[0] += -0.015721142;
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 254))) {
            result[0] += 0.039080422;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 222))) {
              result[0] += 0.020187657;
            } else {
              result[0] += 0.0016309306;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 224))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
            result[0] += -0.01591835;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 64))) {
              result[0] += -0.006518031;
            } else {
              result[0] += -0.0026554007;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 248))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
              result[0] += 0.00103551;
            } else {
              result[0] += 0.00901093;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
              result[0] += -0.0012550136;
            } else {
              result[0] += -1.612309e-05;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 76))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 116))) {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 114))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 130))) {
              result[0] += 0.00788109;
            } else {
              result[0] += -0.0018598955;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 60))) {
              result[0] += 0.005137967;
            } else {
              result[0] += -0.0032950975;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 98))) {
            result[0] += -0.042013966;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 72))) {
              result[0] += -0.0046039354;
            } else {
              result[0] += 0.011869277;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 172))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
              result[0] += -0.026546886;
            } else {
              result[0] += 0.009092043;
            }
          } else {
            result[0] += 0.017844815;
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 230))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 102))) {
              result[0] += 0.0068409294;
            } else {
              result[0] += 0.01327271;
            }
          } else {
            result[0] += 0.000659167;
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 40))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 84))) {
            result[0] += 0.004663289;
          } else {
            result[0] += -0.006825484;
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 146))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 56))) {
              result[0] += 0.008568982;
            } else {
              result[0] += 0.0030761992;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 150))) {
              result[0] += -0.014622196;
            } else {
              result[0] += 0.00041277093;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 42))) {
              result[0] += 0.0016062713;
            } else {
              result[0] += 0.046605274;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 172))) {
              result[0] += -0.004711694;
            } else {
              result[0] += -0.0015061334;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 54))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 222))) {
              result[0] += 0.012894141;
            } else {
              result[0] += -0.0020437974;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 124))) {
              result[0] += -0.0009532698;
            } else {
              result[0] += 0.00017820772;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 274))) {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 120))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += 0.0047425376;
            } else {
              result[0] += 0.009442864;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 58))) {
              result[0] += -0.016419962;
            } else {
              result[0] += 0.00078140077;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 218))) {
              result[0] += -0.016464185;
            } else {
              result[0] += -0.0012191174;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 106))) {
              result[0] += 0.00046863809;
            } else {
              result[0] += 0.0035006986;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 124))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 160))) {
            result[0] += -0.0128822;
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 170))) {
              result[0] += -0.0021174157;
            } else {
              result[0] += -0.01799887;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 80))) {
              result[0] += 0.043405965;
            } else {
              result[0] += 0.0019986776;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.00859444;
            } else {
              result[0] += 0.00027344722;
            }
          }
        }
      }
    } else {
      result[0] += 0.013666634;
    }
  } else {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 88))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 70))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 228))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 0))) {
              result[0] += 0.0035516606;
            } else {
              result[0] += -1.4989321e-05;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 230))) {
              result[0] += -0.003034349;
            } else {
              result[0] += -0.0006231525;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 34))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 244))) {
              result[0] += 0.0052153706;
            } else {
              result[0] += -0.0051266435;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 84))) {
              result[0] += -0.0033818122;
            } else {
              result[0] += -0.0005771067;
            }
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 30))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 244))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
              result[0] += 0.042181622;
            } else {
              result[0] += 0.004492736;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
              result[0] += -0.01460122;
            } else {
              result[0] += 0.00014612076;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 146))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 92))) {
              result[0] += -0.001969084;
            } else {
              result[0] += -6.331263e-05;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 160))) {
              result[0] += 0.0028343403;
            } else {
              result[0] += 0.00031744837;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
        result[0] += -0.006944129;
      } else {
        result[0] += -0.0027726921;
      }
    }
  }
  if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 278))) {
    if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 246))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 232))) {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 66))) {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 60))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
              result[0] += 0.0005400666;
            } else {
              result[0] += -0.0019886708;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
              result[0] += -0.009072294;
            } else {
              result[0] += -0.0026447035;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 76))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
              result[0] += -0.0027691973;
            } else {
              result[0] += 0.00404422;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 276))) {
              result[0] += -2.3040471e-05;
            } else {
              result[0] += 0.012937434;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 22))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
            result[0] += -0.015312892;
          } else {
            result[0] += -0.004679353;
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 102))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
              result[0] += 0.006205307;
            } else {
              result[0] += 0.0010604822;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
              result[0] += -0.0003646382;
            } else {
              result[0] += 0.0056393673;
            }
          }
        }
      }
    } else {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 246))) {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 98))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
            result[0] += 0.0009229826;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 262))) {
              result[0] += -0.002368063;
            } else {
              result[0] += -0.00024388141;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 250))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
              result[0] += -0.012024042;
            } else {
              result[0] += -0.0031967615;
            }
          } else {
            result[0] += -0.00085695874;
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 260))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 204))) {
              result[0] += -0.00046735056;
            } else {
              result[0] += 0.0040739644;
            }
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 258))) {
              result[0] += -0.0032260884;
            } else {
              result[0] += 0.0010031442;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
            result[0] += -0.009778283;
          } else {
            result[0] += 0.019902207;
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 120))) {
      if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 282))) {
        result[0] += -0.0005751638;
      } else {
        result[0] += -0.01039412;
      }
    } else {
      if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 230))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 122))) {
          result[0] += -0.01584022;
        } else {
          result[0] += 0.0018113088;
        }
      } else {
        result[0] += 0.0046630637;
      }
    }
  }
  if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 24))) {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 250))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 4))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 36))) {
              result[0] += -0.00047560144;
            } else {
              result[0] += 0.0013354685;
            }
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 90))) {
              result[0] += -0.0026048406;
            } else {
              result[0] += 0.00038235518;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 256))) {
              result[0] += 0.0044472557;
            } else {
              result[0] += 0.03397901;
            }
          } else {
            result[0] += -0.0003922035;
          }
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 114))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 14))) {
            result[0] += 0.031751767;
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 100))) {
              result[0] += -0.0036668777;
            } else {
              result[0] += 0.00452039;
            }
          }
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 22))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 18))) {
              result[0] += -0.002395912;
            } else {
              result[0] += 0.030751167;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 42))) {
              result[0] += -0.015407211;
            } else {
              result[0] += 0.011755302;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 32))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 6))) {
            result[0] += -0.03115373;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 8))) {
              result[0] += 0.0073042098;
            } else {
              result[0] += 0.033008788;
            }
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 94))) {
            result[0] += -0.0064700614;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 4))) {
              result[0] += -0.007395641;
            } else {
              result[0] += -0.02954722;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 14))) {
          result[0] += -0.01630216;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 184))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 32))) {
              result[0] += 0.008391211;
            } else {
              result[0] += -0.0021036936;
            }
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 62))) {
              result[0] += -0.00060320273;
            } else {
              result[0] += 0.0022427232;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 60))) {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 24))) {
        if (LIKELY((data[6].missing != -1) && (data[6].qvalue < 6))) {
          result[0] += -0.000107325846;
        } else {
          result[0] += 0.029023886;
        }
      } else {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 48))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 50))) {
            result[0] += -0.003524834;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 20))) {
              result[0] += 0.002008538;
            } else {
              result[0] += 0.00040226732;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 132))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 164))) {
              result[0] += 0.0028264725;
            } else {
              result[0] += 0.0078762965;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 190))) {
              result[0] += -0.0036010318;
            } else {
              result[0] += -0.020989217;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 10))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 76))) {
          result[0] += -0.025969082;
        } else {
          if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 76))) {
            result[0] += -0.0037539073;
          } else {
            result[0] += -0.015046534;
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 52))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 46))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
              result[0] += 0.008154599;
            } else {
              result[0] += -0.00034012404;
            }
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.00019781369;
            } else {
              result[0] += -0.0046178605;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 60))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 122))) {
              result[0] += 0.0022514695;
            } else {
              result[0] += 0.02182677;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 66))) {
              result[0] += -0.002138548;
            } else {
              result[0] += 0.00019392448;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 2))) {
    if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 4))) {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 4))) {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 2))) {
          result[0] += -0.018533966;
        } else {
          result[0] += 0.009372647;
        }
      } else {
        result[0] += -0.013326846;
      }
    } else {
      if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 0))) {
        result[0] += 0.017056702;
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 250))) {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 260))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 36))) {
              result[0] += 0.0010617184;
            } else {
              result[0] += 0.003547955;
            }
          } else {
            result[0] += -0.002409021;
          }
        } else {
          result[0] += 0.022888973;
        }
      }
    }
  } else {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 178))) {
      if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 218))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 240))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 238))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 176))) {
              result[0] += -6.381352e-05;
            } else {
              result[0] += 0.01608517;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
              result[0] += 0.031114805;
            } else {
              result[0] += 0.0031871856;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
              result[0] += 0.0010918784;
            } else {
              result[0] += 0.010223559;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
              result[0] += -0.0053569055;
            } else {
              result[0] += -0.00082102185;
            }
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 140))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 122))) {
            result[0] += 0.017271513;
          } else {
            result[0] += -0.0010735758;
          }
        } else {
          if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 210))) {
            result[0] += -0.0022431507;
          } else {
            result[0] += -0.0044196337;
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 114))) {
        result[0] += 0.03313246;
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 198))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 166))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 116))) {
              result[0] += 0.0011855942;
            } else {
              result[0] += 0.0060151634;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 142))) {
              result[0] += -0.007734927;
            } else {
              result[0] += 0.0012392174;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 200))) {
            result[0] += -0.0054051043;
          } else {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 252))) {
              result[0] += -0.00018920467;
            } else {
              result[0] += 0.0010806855;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 270))) {
    if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 210))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 186))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 184))) {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 174))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 170))) {
              result[0] += -5.627934e-05;
            } else {
              result[0] += 0.004937595;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 176))) {
              result[0] += -0.0081005385;
            } else {
              result[0] += -0.00034580208;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 214))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 112))) {
              result[0] += 0.0017502941;
            } else {
              result[0] += 0.0067032403;
            }
          } else {
            result[0] += 0.05561105;
          }
        }
      } else {
        if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 114))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 148))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 104))) {
              result[0] += -0.017684242;
            } else {
              result[0] += -0.008623506;
            }
          } else {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 210))) {
              result[0] += -0.0030512868;
            } else {
              result[0] += 0.00268173;
            }
          }
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 138))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 116))) {
              result[0] += -0.008280813;
            } else {
              result[0] += -0.0006436382;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 68))) {
              result[0] += 0.005869223;
            } else {
              result[0] += 0.0157161;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 218))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 38))) {
          result[0] += -0.040014755;
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 68))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 64))) {
              result[0] += 0.0034462882;
            } else {
              result[0] += 0.012366778;
            }
          } else {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 126))) {
              result[0] += 0.0029618521;
            } else {
              result[0] += -0.0031078772;
            }
          }
        }
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 78))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 68))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 156))) {
              result[0] += -0.0013473729;
            } else {
              result[0] += 0.0062149167;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
              result[0] += 0.017057097;
            } else {
              result[0] += -0.010900224;
            }
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 144))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 212))) {
              result[0] += 0.0012661046;
            } else {
              result[0] += 0.0069153532;
            }
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
              result[0] += -0.016097846;
            } else {
              result[0] += -0.00017735653;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 0))) {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 272))) {
        result[0] += 0.010808589;
      } else {
        if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 218))) {
          result[0] += -0.009866092;
        } else {
          result[0] += 0.0047606938;
        }
      }
    } else {
      if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 216))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 108))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 266))) {
            result[0] += -0;
          } else {
            result[0] += 0.014268956;
          }
        } else {
          result[0] += 0.0023136109;
        }
      } else {
        result[0] += -0.0024249533;
      }
    }
  }
  if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 96))) {
    if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 74))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 64))) {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 60))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 40))) {
            if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 136))) {
              result[0] += -0.0002939115;
            } else {
              result[0] += 0.03095895;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 36))) {
              result[0] += -0.016124493;
            } else {
              result[0] += 0.0012605732;
            }
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 128))) {
            result[0] += 0.00055091607;
          } else {
            result[0] += -0.0038312187;
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 62))) {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
            result[0] += 0.020991528;
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 30))) {
              result[0] += 0.00425994;
            } else {
              result[0] += -0.012923174;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 72))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 112))) {
              result[0] += -0.008987924;
            } else {
              result[0] += 0.0027231926;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 136))) {
              result[0] += 0.00033489117;
            } else {
              result[0] += 0.0031028683;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 94))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 76))) {
          result[0] += -0.023087258;
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 40))) {
            result[0] += 0.040452894;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 20))) {
              result[0] += -0.004355786;
            } else {
              result[0] += 0.00052141724;
            }
          }
        }
      } else {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 98))) {
          result[0] += -0.026337389;
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 78))) {
            result[0] += -0.011440942;
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 80))) {
              result[0] += 0.0051072245;
            } else {
              result[0] += -0.0016136405;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 94))) {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 104))) {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 66))) {
            result[0] += -0.016175192;
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 90))) {
              result[0] += -0.003506745;
            } else {
              result[0] += -0.0008637124;
            }
          }
        } else {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 164))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 80))) {
              result[0] += 0.039352115;
            } else {
              result[0] += 0.009527031;
            }
          } else {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 168))) {
              result[0] += -0.014133029;
            } else {
              result[0] += 0.0009379142;
            }
          }
        }
      } else {
        if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 124))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
            result[0] += 0.0014655776;
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 94))) {
              result[0] += 0.00799122;
            } else {
              result[0] += 0.02205122;
            }
          }
        } else {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 150))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 116))) {
              result[0] += 0.00084181456;
            } else {
              result[0] += 0.03656875;
            }
          } else {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
              result[0] += -0.0324972;
            } else {
              result[0] += -0.0043041403;
            }
          }
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 108))) {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 130))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 106))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 120))) {
              result[0] += -0.014786902;
            } else {
              result[0] += -0.027247995;
            }
          } else {
            result[0] += -0.013442961;
          }
        } else {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 106))) {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 96))) {
              result[0] += 0.014649215;
            } else {
              result[0] += 0.00018173472;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 82))) {
              result[0] += -0.003195444;
            } else {
              result[0] += -0.017632857;
            }
          }
        }
      } else {
        if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 218))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 172))) {
            if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 180))) {
              result[0] += 0.0011816436;
            } else {
              result[0] += 0.006784923;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 174))) {
              result[0] += -0.0025611892;
            } else {
              result[0] += 0.00039800836;
            }
          }
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 222))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 108))) {
              result[0] += -0.015747108;
            } else {
              result[0] += -0.0025709863;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 222))) {
              result[0] += 0.00016036122;
            } else {
              result[0] += -0.004456304;
            }
          }
        }
      }
    }
  }
  if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 116))) {
    if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 44))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 0))) {
        result[0] += -0.043082695;
      } else {
        if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 2))) {
          if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 14))) {
            result[0] += -0.015321686;
          } else {
            if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 18))) {
              result[0] += 0.009543039;
            } else {
              result[0] += -0.00492233;
            }
          }
        } else {
          if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 60))) {
            result[0] += -0.018702324;
          } else {
            result[0] += 0.0030421054;
          }
        }
      }
    } else {
      if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 110))) {
        if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 96))) {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 104))) {
              result[0] += 0.023666665;
            } else {
              result[0] += -0.003085631;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 102))) {
              result[0] += 0.0025167426;
            } else {
              result[0] += -0.0006228736;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 126))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 52))) {
              result[0] += 0.025037153;
            } else {
              result[0] += 0.00086976105;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 130))) {
              result[0] += -0.00833582;
            } else {
              result[0] += 6.457727e-05;
            }
          }
        }
      } else {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 48))) {
          result[0] += -0.034489177;
        } else {
          if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 50))) {
            result[0] += 0.059084423;
          } else {
            if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 54))) {
              result[0] += -0.008554864;
            } else {
              result[0] += -0.000933964;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 98))) {
      if (UNLIKELY((data[6].missing != -1) && (data[6].qvalue < 2))) {
        result[0] += -0.005567677;
      } else {
        if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 74))) {
          result[0] += 0.026860246;
        } else {
          if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 240))) {
            if (LIKELY((data[1].missing != -1) && (data[1].qvalue < 238))) {
              result[0] += 0.003906974;
            } else {
              result[0] += 0.016403912;
            }
          } else {
            result[0] += -0.0005779523;
          }
        }
      }
    } else {
      if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 102))) {
        if (UNLIKELY((data[1].missing != -1) && (data[1].qvalue < 68))) {
          result[0] += -0.021588845;
        } else {
          result[0] += -0.0072186044;
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 152))) {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 88))) {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 56))) {
              result[0] += 0.00058102736;
            } else {
              result[0] += 0.0042956616;
            }
          } else {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
              result[0] += -0.0021127523;
            } else {
              result[0] += 0.0012841078;
            }
          }
        } else {
          if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 162))) {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 186))) {
              result[0] += -0.004427323;
            } else {
              result[0] += -0.0004450692;
            }
          } else {
            if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 188))) {
              result[0] += 0.0005094443;
            } else {
              result[0] += -0.00016178952;
            }
          }
        }
      }
    }
  }
  if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 162))) {
    if (LIKELY((data[4].missing != -1) && (data[4].qvalue < 152))) {
      if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 232))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 230))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 272))) {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 50))) {
              result[0] += -0.00010068305;
            } else {
              result[0] += 0.0004361026;
            }
          } else {
            result[0] += 0.027371911;
          }
        } else {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 130))) {
            result[0] += 0.023174508;
          } else {
            result[0] += 0.011960031;
          }
        }
      } else {
        if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 242))) {
          if (LIKELY((data[2].missing != -1) && (data[2].qvalue < 100))) {
            if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 62))) {
              result[0] += -0.015924754;
            } else {
              result[0] += -0.0012669711;
            }
          } else {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 188))) {
              result[0] += 0.018523319;
            } else {
              result[0] += -0.00022792198;
            }
          }
        } else {
          result[0] += 0.039226815;
        }
      }
    } else {
      if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 112))) {
        if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 146))) {
          if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 102))) {
            if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 90))) {
              result[0] += 0.022155268;
            } else {
              result[0] += -0.0030770514;
            }
          } else {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 66))) {
              result[0] += 0.03438976;
            } else {
              result[0] += 0.016168883;
            }
          }
        } else {
          if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 80))) {
            if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 128))) {
              result[0] += 0.023656188;
            } else {
              result[0] += 0.009587833;
            }
          } else {
            if (LIKELY((data[7].missing != -1) && (data[7].qvalue < 118))) {
              result[0] += 0.003995668;
            } else {
              result[0] += -0.00010666515;
            }
          }
        }
      } else {
        if (UNLIKELY((data[0].missing != -1) && (data[0].qvalue < 108))) {
          if (UNLIKELY((data[7].missing != -1) && (data[7].qvalue < 150))) {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 136))) {
              result[0] += 0.004109849;
            } else {
              result[0] += 0.027041653;
            }
          } else {
            result[0] += -9.42766e-05;
          }
        } else {
          if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 154))) {
            result[0] += 0.0130727235;
          } else {
            if (LIKELY((data[3].missing != -1) && (data[3].qvalue < 78))) {
              result[0] += -0.0015822932;
            } else {
              result[0] += -0.008224895;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 164))) {
      result[0] += -0.012683782;
    } else {
      if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 38))) {
        if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 268))) {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 146))) {
            result[0] += -0.0033396825;
          } else {
            if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 162))) {
              result[0] += 0.01649657;
            } else {
              result[0] += 0.0024240483;
            }
          }
        } else {
          if (UNLIKELY((data[2].missing != -1) && (data[2].qvalue < 186))) {
            result[0] += -0.004931916;
          } else {
            result[0] += 4.3331627e-05;
          }
        }
      } else {
        if (UNLIKELY((data[3].missing != -1) && (data[3].qvalue < 52))) {
          if (LIKELY((data[0].missing != -1) && (data[0].qvalue < 202))) {
            if (UNLIKELY((data[5].missing != -1) && (data[5].qvalue < 254))) {
              result[0] += -0.0077444026;
            } else {
              result[0] += -0.0008859424;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 178))) {
              result[0] += 0.0073036547;
            } else {
              result[0] += -0.007702441;
            }
          }
        } else {
          if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 228))) {
            if (LIKELY((data[5].missing != -1) && (data[5].qvalue < 220))) {
              result[0] += -0.00019439656;
            } else {
              result[0] += -0.0056553436;
            }
          } else {
            if (UNLIKELY((data[4].missing != -1) && (data[4].qvalue < 180))) {
              result[0] += 0.0066337893;
            } else {
              result[0] += 0.00068436674;
            }
          }
        }
      }
    }
  }

  // Apply base_scores
  result[0] += 3.356329679489135742;

  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void pdlp_predictor::postprocess(double* result)
{
  // Do nothing
}

// Feature names array
const char* pdlp_predictor::feature_names[pdlp_predictor::NUM_FEATURES] = {
  "n_vars", "n_cstrs", "nnz", "sparsity", "nnz_stddev", "unbalancedness", "spmv_ops", "total_nnz"};
