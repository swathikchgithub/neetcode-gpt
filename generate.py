import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        #torch.manual_seed(0)
        #initial_state = generator.get_state()
        
        if isinstance(context, torch.Tensor):
            idx = context.clone().detach().to(torch.long)
        else:
            idx = torch.tensor(context, dtype=torch.long)
            
        # 2. ఆటోరెగ్రెసివ్ జనరేషన్ లూప్
        for _ in range(new_chars):
            # కంటెక్స్ట్ లెంత్ క్రాపింగ్
            idx_cond = idx[:, -context_length:]
            
            # ఫార్వర్డ్ పాస్
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # సాఫ్ట్‌మాక్స్ మరియు మల్టీనోమియల్ శాంప్లింగ్
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            
            # కాంటెక్స్ట్‌కి అపెండ్ చేయడం
            idx = torch.cat((idx, next_index), dim=1)
            
        # 3. సురక్షితంగా డిక్షనరీ నుండి క్యారెక్టర్లను డీకోడ్ చేయడం
        tokens = idx[0, -new_chars:].tolist()
        result_text = "".join([int_to_char.get(t, "") for t in tokens])
        
        return result_text
        
        
        
                # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
