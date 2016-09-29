package linkedIn;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

public class Person {
	Person[] parents;

	public Person() {
		parents = new Person[2];
	}

	public boolean bloodRelated(Person p1, Person p2) {
		if (p1 == p2)
			return true;
		HashSet<Person> p1Ancestors = new HashSet<Person>();
		HashSet<Person> p2Ancestors = new HashSet<Person>();

		List<Person> p1Discovered = new LinkedList<Person>();
		p1Discovered.add(p1);
		List<Person> p2Discovered = new LinkedList<Person>();
		p2Discovered.add(p2);

		while (!p1Discovered.isEmpty() || !p2Discovered.isEmpty()) {
			Person nextP1 = p1Discovered.remove(0);
			if (nextP1 != null) {
				if (p2Ancestors.contains(nextP1)) {
					return true;
				}

				for (Person parent : nextP1.parents) {
					p1Discovered.add(parent);
				}
				p1Ancestors.add(nextP1);
			}

			Person nextP2 = p2Discovered.remove(0);
			if (nextP2 != null) {
				if (p1Ancestors.contains(nextP2)) {
					return true;
				}

				for (Person parent : nextP2.parents) {
					p2Discovered.add(parent);
				}
				p2Ancestors.add(nextP2);
			}
		}
		return false;
	}
}
