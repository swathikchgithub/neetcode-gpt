from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        words = corpus.split()
        splits = [list(word) for word in words]
        merges = []

        for _ in range(num_merges):
            pair_counts = defaultdict(int)
            for split in splits:
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    pair_counts[pair] += 1
            
            if not pair_counts:
                break

            best_pair = min(pair_counts.keys(), key=lambda p:(-pair_counts[p],p))
            merges.append(list(best_pair))

            new_splits = []
            for split in splits:
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                        new_split.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_split.append(split[i])
                        i+=1
                new_splits.append(new_split)
            splits = new_splits
        return merges


