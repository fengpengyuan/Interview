package com.leetcode2;

import java.util.HashMap;
import java.util.Map;

public class TwoSum {
	Map<Integer, Integer> map = new HashMap<Integer, Integer>();

	public void add(int number) {
		if (map.containsKey(number))
			map.put(number, map.get(number) + 1);
		else
			map.put(number, 1);
	}

	public boolean find(int value) {
		for (int key : map.keySet()) {
			int left = value - key;
			if (map.containsKey(left)) {
				if (left == key && map.get(key) < 2)
					continue;
				return true;
			}
		}
		return false;
	}
}
