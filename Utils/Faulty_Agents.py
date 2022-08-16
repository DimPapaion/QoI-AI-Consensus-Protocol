import numpy as np
import random
import torch
import torch.nn.functional as F
from keras.api._v2.keras.utils import to_categorical


def generate_faulty_agent(classes, samples):
  target_faulty = np.array([random.randrange(classes)for i in range(samples)])
  target_OHE_Faulty = to_categorical(target_faulty)
  print(type(target_OHE_Faulty))
  for i in range(target_OHE_Faulty.shape[0]):
    for j in range(target_OHE_Faulty.shape[1]):
      if target_OHE_Faulty[i][j] == 1.0:
        target_OHE_Faulty[i][j] = 0.89
      else:
        target_OHE_Faulty[i][j] = 1/10#random.uniform(0.01, 0.1)

  return target_OHE_Faulty, target_faulty

def test_faulty(args, preds_total):
    preds_total_soft = torch.Tensor(preds_total)
    preds_total_ = F.softmax(preds_total_soft, dim=1)

    faulty_1_OHE, faulty_1 = generate_faulty_agent(classes=args.n_classes, samples=args.n_samples)
    faulty_2_OHE, faulty_2 = generate_faulty_agent(classes=args.n_classes, samples=args.n_samples)
    faulty_3_OHE, faulty_3 = generate_faulty_agent(classes=args.n_classes, samples=args.n_samples)
    faulty_4_OHE, faulty_4 = generate_faulty_agent(classes=args.n_classes, samples=args.n_samples)
    faulty_5_OHE, faulty_5 = generate_faulty_agent(classes=args.n_classes, samples=args.n_samples)
    faulty_6_OHE, faulty_6 = generate_faulty_agent(classes=args.n_classes, samples=args.n_samples)

    preds = (np.stack(preds_total_)).tolist()
    preds = np.stack(preds)

    preds = (np.stack(preds)).tolist()
    preds.pop(5)
    preds.append(faulty_5_OHE)
    preds = np.stack(preds)

    return preds