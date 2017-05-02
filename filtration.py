### Sublevel filtration

def sortX(points):
	retPoints = sorted(points, key=lambda point: point[0])
	return retPoints

def sortY(points):
	retPoints = sorted(points, key=lambda point: point[1])
	return retPoints

def makeEdges(points, pti):
	## Sort by x to make edges
	## Edge tuplet is point indices (e.g. edge between point 3 and 6 is (3,6))
	points = sortX(points)

	edges = []
	prevPt = points[0]
	for x in range(1, len(points)):
		edges.append((pti[prevPt],pti[points[x]]))
		prevPt = points[x]

	return edges

def getEdgeWeight(edge):
	p0 = edge[0]
	p1 = edge[1]
	if p1[1] > p0[1]:
		return p1[1]
	else:
		return p0[1]

def makeCompList(points):
	## Create list of all components
	comps = []

	## Sort points by y index, if they haven't been sorted already
	points = sortY(points)

	## Add points first -- record index to use for edges
	pointToIndex = {}
	indexToPoint = {}

	## Field 0 is type, Field 1 is weight, Field 2 is the point/edge tuplet, Field 3 is index (for points)
	## type -- 0 means point, 1 means edge
	index = 0
	for point in points:
		comps.append((0, point[1], point, index))
		pointToIndex[point] = index
		indexToPoint[index] = point
		index += 1

	edges = makeEdges(points, pointToIndex)

	for edge in edges:
		comps.append((1, getEdgeWeight((indexToPoint[edge[0]], indexToPoint[edge[1]])), edge, -1))

	## Sort the components list by weight, then by type so points are added before edges at same height
	return sorted(comps, key = lambda weight: (weight[1], weight[0]))

def makeHeadArray(comps):
	## Make component head array -- indices are the point indices

	## Heads are sorted in ascending order by weight -- a higher index means later addition
	heads = []
	for x in range(0, len(comps)):
		if comps[x][0] == 0: ## Is a point, not an edge
			heads.append(comps[x][3]) ## The index of the point

	return heads

def pruneHead(p, heads, pSet):
	## Recurses until it finds an index that points to itself, then points all points in the chain to that point
	if heads[p] == p:
		if pSet != []:
			for pAdd in pSet:
				heads[pAdd] = p
		return p
	else:
		pSet.append(p)
		return pruneHead (heads[p], heads, pSet)

#### Main function
def filter(points):

	realpoints = []
	for i in range(len(points)):
		realpoints.append((i,points[i]))

	comps = makeCompList(realpoints)

	heads = makeHeadArray(comps)

	## Points in the persistence diagram
	perPoints = []

	## A point index maps to its birth -- look this up again to kill it and make the persistence point
	birthMap = {}

	for x in range(0, len(comps)):
		comp = comps[x]
		if comp[0] == 0:
			## Point -- Map the component to its birth weight
			birthMap[pruneHead(comp[3], heads, [])] = comp[1]
			# print("Birth: ", comp[2])
		else:
			## Edge -- Kill something
				p1 = comp[2][0]
				p2 = comp[2][1]
				head1 = pruneHead(p1, heads, [])
				head2 = pruneHead(p2, heads, [])
				if head1 != head2:
					if head1 > head2:
						high = head1
						hp = p1
						low = head2
					else:
						high = head2
						hp = p2
						low = head1
					## Kill higher component

					## First, add the persistence point (if it's not on the diagonal)
					# print("Death: ", comp[1])
					# print (birthMap[hp])
					if birthMap[high] != comp[1]:
						perPoints.append((birthMap[high], comp[1]))

					## Then, point the higher component head to the lower one -- now they are merged
					heads[high] = low

	return perPoints

###TEST
# sample_in = [(1,2),(2,4),(3,8),(4,3),(5,6),(6,2),(7,7)]
# print(filter(sample_in))
