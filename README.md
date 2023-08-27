# from_scratch
This repo will implement algorithms from scratch. The objective is to learn more about the alogrithms.

The objective is also to get better at go.

#Algo: Hyperloglog

Say you want to count the number of unique items in a set. 

Approach 1: A stack with unique items. When a new item is added it is checked across the stack to see if the same elemnet is already present. Time Complexity : O(n^2), Space Complexity: O(n). In case we have a trillion unique items the time complexity will blow up.

Approach 2: Let's solve the time complexity issue. Hashmap. Each new element takes O(1). n elements time complexity O(n), space complexity: O(n)

Both of these are accurate measures. When you have to trade-off some accuracy to get better at time-complexity

Each unique item mapped to a binary number: For example say 100111000
1. Take the length of the longest run of 0s from the end. Set the counter to that length.
2. Next Item: Find the unique binary numebr corresponding to it. Take the length of 0s. If length >current length set counter to length.
3. Keep doing this. No space complexity O(1), Time Complexity: O(n)


Corner Cases: First Bin encoding is say 110000000: So counter set to a high number already.

To solve this: Take first bits to decide on the box to send the length to. Then take harmonic mean of the output of the boxes.
