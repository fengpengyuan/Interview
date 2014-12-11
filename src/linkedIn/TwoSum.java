package linkedIn;

import java.util.Hashtable;
import java.util.Iterator;

public class TwoSum {
	public void save(int input) {
		int originalCount = 0;
		if (h.containsKey(input)) {
			originalCount = (int) h.get(input);
		}
		h.put(input, originalCount + 1);
	}

	public boolean test(int test) {

		Iterator<Integer> i = h.keySet().iterator();

		while (i.hasNext()) {
			int c = i.next();

			if (h.containsKey(test - c)) {
				boolean isDouble = test == c * 2;
				int frequency = (int) h.get(c);
				boolean appearOnlyOnce = (frequency == 1);
				if (!(isDouble && appearOnlyOnce))
					return true;
			}
		}

		return false;
	}

	Hashtable<Integer,Integer> h = new Hashtable<Integer, Integer>();
}
