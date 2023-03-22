import numpy as np
import copy

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Crossover():
    @staticmethod
    def any(game1, game2):
        return Crossover.cycle(game1, game2)
        # if np.random.uniform(0, 1) <= 0.5:
        #     return Crossover.cycle(game1, game2)
        # return Crossover.partially_mapped(game1, game2)
        # return Crossover.retain_relationship(game1, game2)
   
    @staticmethod
    def cycle(game1, game2, cnt=0):
        n = game1.n
        child1 = copy.deepcopy(game1)
        child2 = copy.deepcopy(game2)

        crossover_p1, crossover_p2 = np.random.randint(1, n + 2), np.random.randint(1, n + 2) 
        while crossover_p1 == crossover_p2:
            crossover_p1, crossover_p2 = np.random.randint(1, n + 2), np.random.randint(1, n + 2) 
        if crossover_p1 > crossover_p2:
            crossover_p1, crossover_p2 = crossover_p2, crossover_p1

        success = False
        for r in range(crossover_p1, crossover_p2):
            c1_r, c2_r = Crossover.cycle_row(child1.board[r, 1:n + 1], child2.board[r, 1:n + 1])
            if (Game.satisfy_possible(c1_r, r, child1.possible, n) and Game.satisfy_possible(c2_r, r, child2.possible, n) and
                np.any(child1.board[r, 1:n + 1] != c1_r) and np.any(child2.board[r, 1:n + 1] != c2_r)):
                child1.board[r, 1:n + 1], child2.board[r, 1:n + 1] = c1_r, c2_r
                success = True
        if not success and cnt < 20:
            return Crossover.cycle(game1, game2, cnt + 1)

        child1.update_fitness()
        child2.update_fitness()
        return child1, child2


    @staticmethod
    def cycle_row(r1, r2):
        n = len(r1)
        remaining = list(range(1, n + 1))

        child_r1 = np.zeros_like(r1)
        child_r2 = np.zeros_like(r2)

        even = False
        while (0 in child_r1) and (0 in child_r2):
            idx = Crossover.find_unused(r1, remaining)
            start = r1[idx]
            remaining.remove(r1[idx])
            if even:
                child_r1[idx], child_r2[idx] = r1[idx], r2[idx]
            else:
                child_r1[idx], child_r2[idx] = r2[idx], r1[idx]

            # cur = r2[idx]
            while (cur := r2[idx]) != start:
                idx = Crossover.find_next(r1, cur)
                remaining.remove(r1[idx])
                if even:
                    child_r1[idx], child_r2[idx] = r1[idx], r2[idx]
                else:
                    child_r1[idx], child_r2[idx] = r2[idx], r1[idx]

            even = not even
        return child_r1, child_r2

    @staticmethod
    def find_unused(r1, remaining):
        n = len(r1)
        for i in range(n):
            if r1[i] in remaining:
                return i

    @staticmethod
    def find_next(r1, cur):
        n = len(r1)
        for i in range(n):
            if r1[i] == cur:
                return i

    @staticmethod
    def partially_mapped(game1, game2, cnt=0):
        n = game1.n
        child1 = copy.deepcopy(game1)
        child2 = copy.deepcopy(game2)

        crossover_p1, crossover_p2 = np.random.randint(1, n + 2), np.random.randint(1, n + 2) 
        while crossover_p1 == crossover_p2:
            crossover_p1, crossover_p2 = np.random.randint(1, n + 2), np.random.randint(1, n + 2) 
        if crossover_p1 > crossover_p2:
            crossover_p1, crossover_p2 = crossover_p2, crossover_p1

        success = False
        for r in range(crossover_p1, crossover_p2):
            c1_r, c2_r = Crossover.partially_mapped_row(child1.board[r, 1:n + 1], child2.board[r, 1:n + 1])
            if (Game.satisfy_possible(c1_r, r, child1.possible, n) and Game.satisfy_possible(c2_r, r, child2.possible, n) and
                np.any(child1.board[r, 1:n + 1] != c1_r) and np.any(child2.board[r, 1:n + 1] != c2_r)):
                child1.board[r, 1:n + 1], child2.board[r, 1:n + 1] = c1_r, c2_r
                success = True
        if not success and cnt < 20:
            return Crossover.partially_mapped(game1, game2, cnt + 1)

        child1.update_fitness()
        child2.update_fitness()
        return child1, child2

    @staticmethod
    def partially_mapped_row(parent1, parent2):
        cutoff_1, cutoff_2 = np.sort(np.random.choice(np.arange(len(parent1)+1), size=2, replace=False))
        child_r1 = Crossover.partially_mapped_one_offspring(parent1, parent2, cutoff_1, cutoff_2)
        child_r2 = Crossover.partially_mapped_one_offspring(parent2, parent1, cutoff_1, cutoff_2)
        return child_r1, child_r2

    @staticmethod
    def partially_mapped_one_offspring(p1, p2, cutoff_1, cutoff_2):
        offspring = np.zeros(len(p1), dtype=p1.dtype)

        # Copy the mapping section (middle) from parent1
        offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

        # copy the rest from parent2 (provided it's not already there
        for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
            candidate = p2[i]
            while candidate in p1[cutoff_1:cutoff_2]: # allows for several successive mappings
                candidate = p2[np.where(p1 == candidate)[0][0]]
            offspring[i] = candidate
        return offspring

    @staticmethod
    def retain_relationship(game1, game2, cnt=0):
        n = game1.n
        child1 = copy.deepcopy(game1)
        child2 = copy.deepcopy(game2)

        crossover_p1, crossover_p2 = np.random.randint(1, n + 2), np.random.randint(1, n + 2) 
        while crossover_p1 == crossover_p2:
            crossover_p1, crossover_p2 = np.random.randint(1, n + 2), np.random.randint(1, n + 2) 
        if crossover_p1 > crossover_p2:
            crossover_p1, crossover_p2 = crossover_p2, crossover_p1

        success = False
        for r in range(crossover_p1, crossover_p2):
            c1_r, c2_r = Crossover.retain_relationship_row(child1.board[r, 1:n + 1], child2.board[r, 1:n + 1])
            if (Game.satisfy_possible(c1_r, r, child1.possible, n) and Game.satisfy_possible(c2_r, r, child2.possible, n) and
                np.any(child1.board[r, 1:n + 1] != c1_r) and np.any(child2.board[r, 1:n + 1] != c2_r)):
                child1.board[r, 1:n + 1], child2.board[r, 1:n + 1] = c1_r, c2_r
                success = True
        if not success and cnt < 20:
            return Crossover.partially_mapped(game1, game2, cnt + 1)

        child1.update_fitness()
        child2.update_fitness()
        return child1, child2

    @staticmethod
    def retain_relationship_row(r1, r2):
        n = len(r1)

        child_r1 = np.copy(r1)
        child_r2 = np.copy(r2)
        if np.random.uniform(0, 1) <= 0.5:
            p1 = np.random.randint(0, n - 1)
            child_r1[p1:], child_r2[p1:] = Crossover.retain_relationship_offspring(child_r1[p1:], child_r2[p1:])
        else:
            p2 = np.random.randint(2, n + 1)
            child_r1[:p2], child_r2[:p2] = Crossover.retain_relationship_offspring(child_r1[:p2], child_r2[:p2])
        return child_r1, child_r2

    @staticmethod
    def retain_relationship_offspring(r1, r2):
        nums_r1 = np.sort(r1)
        nums_r2 = np.sort(r2)
        mapping_r1_to_r2 = {}
        mapping_r2_to_r1 = {}
        for n1, n2 in zip(nums_r1, nums_r2):
            mapping_r1_to_r2[n1] = n2
            mapping_r2_to_r1[n2] = n1

        child_r1 = np.array([mapping_r1_to_r2[n1] for n1 in r1])
        child_r2 = np.array([mapping_r2_to_r1[n2] for n2 in r2])
        return child_r2, child_r1



class Mutate():
    @staticmethod
    def any(old_game):
        # return Mutate.inverse_relationship(old_game)
        # return Mutate.swap(old_game)
        
        if np.random.normal(0, 1) <= 1 / 2:
            return Mutate.inverse(old_game)
        return Mutate.inverse_relationship(old_game)

        # r = np.random.normal(0, 1)
        # if r < 1 / 3:
        #     return Mutate.swap(old_game)
        # elif r < 2 / 3:
        #     return Mutate.inverse(old_game)
        # return Mutate.inverse_relationship(old_game)


    @staticmethod
    def swap(old_game, cnt=0):
        game = copy.deepcopy(old_game)

        from_c, to_c = np.random.randint(1, game.n + 1), np.random.randint(1, game.n + 1)
        while from_c == to_c:
            from_c, to_c = np.random.randint(1, game.n + 1), np.random.randint(1, game.n + 1)

        mutate_p1, mutate_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2) 
        while mutate_p1 == mutate_p2 or abs(mutate_p1 - mutate_p2) > int(game.n / 3) + 1:
            mutate_p1, mutate_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2) 
        if mutate_p1 > mutate_p2:
            mutate_p1, mutate_p2 = mutate_p2, mutate_p1

        success = False
        for r in range(mutate_p1, mutate_p2):
            if game.board[r, from_c] in game.possible[r][to_c] and game.board[r, to_c] in game.possible[r][from_c]:
                game.board[r, (from_c, to_c)] = game.board[r, (to_c, from_c)]
                success = True
        if not success and cnt < 20:
            return Mutate.swap(old_game, cnt + 1)

        game.update_fitness()
        return game

    @staticmethod
    def inverse(old_game, cnt=0):
        game = copy.deepcopy(old_game)

        mutate_p1, mutate_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2) 
        while mutate_p1 == mutate_p2 or abs(mutate_p1 - mutate_p2) > int(game.n / 3) + 1:
            mutate_p1, mutate_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2) 
        if mutate_p1 > mutate_p2:
            mutate_p1, mutate_p2 = mutate_p2, mutate_p1

        swap_p1, swap_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2)
        while abs(swap_p1 - swap_p2) < 2:
            swap_p1, swap_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2)
        if swap_p1 > swap_p2:
            swap_p1, swap_p2 = swap_p2, swap_p1

        success = False
        for r in range(mutate_p1, mutate_p2):
            new_r = copy.deepcopy(game.board[r])
            new_r[swap_p1:swap_p2] = game.board[r, swap_p2 - 1:swap_p1 - 1:-1]
            if Game.satisfy_possible(new_r[1:game.n + 1], r, game.possible, game.n):
                game.board[r] = new_r
                success = True
        if not success and cnt < 20:
            return Mutate.inverse(old_game, cnt + 1)

        game.update_fitness()
        return game

    @staticmethod
    def inverse_relationship(old_game, cnt=0):
        game = copy.deepcopy(old_game)

        mutate_p1, mutate_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2) 
        while mutate_p1 == mutate_p2 or abs(mutate_p1 - mutate_p2) > int(game.n / 3) + 1:
            mutate_p1, mutate_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2) 
        if mutate_p1 > mutate_p2:
            mutate_p1, mutate_p2 = mutate_p2, mutate_p1

        swap_p1, swap_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2)
        while abs(swap_p1 - swap_p2) < 2:
            swap_p1, swap_p2 = np.random.randint(1, game.n + 2), np.random.randint(1, game.n + 2)
        if swap_p1 > swap_p2:
            swap_p1, swap_p2 = swap_p2, swap_p1

        # print(f"{swap_p1 = }, {swap_p2 = }, {mutate_p1 = }, {mutate_p2 = }")

        success = False
        for r in range(mutate_p1, mutate_p2):
            new_r = copy.deepcopy(game.board[r])
            new_r[swap_p1:swap_p2] = Mutate.inverse_relationship_row(new_r[swap_p1:swap_p2])
            # new_r[swap_p1:swap_p2] = game.board[r, swap_p2 - 1:swap_p1 - 1:-1]
            if Game.satisfy_possible(new_r[1:game.n + 1], r, game.possible, game.n):
                game.board[r] = new_r
                success = True
        if not success and cnt < 20:
            return Mutate.inverse_relationship(old_game, cnt + 1)

        game.update_fitness()
        return game
    
    @staticmethod
    def inverse_relationship_row(row):
        nums = np.sort(row)
        inverse_nums = nums[::-1]
        mapping_nums_to_inverse = {}
        for n1, n2 in zip(nums, inverse_nums):
            mapping_nums_to_inverse[n1] = n2
        res = np.array([mapping_nums_to_inverse[n1] for n1 in row])
        return res


class Game():
    def __init__(self, board):
        self.board = np.array(board)
        self.n = self.board.shape[0] - 2
        self.is_given = (self.board != 0)
        self.restrictions = np.array([self.board[0, 1:self.n + 1], self.board[self.n + 1, 1:self.n + 1],
                                      self.board[1:self.n + 1, 0], self.board[1:self.n + 1, self.n + 1]])
        self.fitness = 0
        self.create_possible()

        # print(self)
        # for r in range(self.n + 2):
        #     for c in range(self.n + 2):
        #         print(f"{self.possible[r][c]}", end=' ')
        #     print("")
        
        self.update_possible()

        # print(self)
        # for r in range(self.n + 2):
        #     for c in range(self.n + 2):
        #         print(f"{self.possible[r][c]}", end=' ')
        #     print("")

        self.create_random_board()
        self.update_fitness()
        
    def create_possible(self):
        self.possible = [[set(range(1, self.n + 1)) for c in range(self.n + 2)] for r in range(self.n + 2)]
        
        for r in range(self.n + 2):
            for c in range(self.n + 2):
                if r == 0 or r == self.n + 1 or c == 0 or c == self.n + 1:
                    self.possible[r][c] = set()
                    # continue
                elif self.board[r, c] != 0:
                    self.possible[r][c] = set([self.board[r, c]])
                    for i in range(1, self.n + 1):
                        if i != r:
                            self.possible[i][c].discard(self.board[r, c])
                        if i != c:
                            self.possible[r][i].discard(self.board[r, c])

    def update_possible(self):
        # heuristic

        for r in range(self.n + 2):
            # pass
            if self.board[r, 0] == 1:
                self.possible[r][1] = set([self.n])
            elif self.board[r, 0] > 1:
                # for i in range(1, self.board[r, 0]):
                #     self.possible[r][i].discard(self.n)
                for minus in range(self.board[r, 0] - 1):
                    for i in range(1, self.board[r, 0] - minus):
                        self.possible[r][i].discard(self.n - minus)
            if self.board[r, 0] == 2:
                self.possible[r][2].discard(self.n - 1)

            if self.board[r, self.n + 1] == 1:
                self.possible[r][self.n] = set([self.n])
            elif self.board[r, self.n + 1] > 1:
                # for i in range(self.n, self.n + 1 - self.board[r, self.n + 1], -1):
                #     self.possible[r][i].discard(self.n)
                for minus in range(self.board[r, self.n + 1] - 1):
                    for i in range(self.n, self.n + 1 - self.board[r, self.n + 1] + minus, -1):
                        self.possible[r][i].discard(self.n - minus)
            if self.board[r, self.n + 1] == 2:
                self.possible[r][self.n - 1].discard(self.n - 1)

        for c in range(self.n + 2):
            if self.board[0, c] == 1:
                self.possible[1][c] = set([self.n])
            elif self.board[0, c] > 1:
                # for i in range(1, self.board[0, c]):
                #     self.possible[i][c].discard(self.n)
                for minus in range(self.board[0, c] - 1):
                    for i in range(1, self.board[0, c] - minus):
                        self.possible[i][c].discard(self.n - minus)
            if self.board[0, c] == 2:
                self.possible[2][c].discard(self.n - 1)

            if self.board[self.n + 1, c] == 1:
                self.possible[self.n][c] = set([self.n])
            elif self.board[self.n + 1, c] > 1:
                # for i in range(self.n, self.n + 1 - self.board[self.n + 1, c], -1):
                #     self.possible[i][c].discard(self.n)
                for minus in range(self.board[self.n + 1, c] - 1):
                    for i in range(self.n, self.n + 1 - self.board[self.n + 1, c] + minus, -1):
                        self.possible[i][c].discard(self.n - minus)
            if self.board[self.n + 1, c] == 2:
                self.possible[self.n - 1][c].discard(self.n - 1)

        update = True
        while update:
            update = False
            for r in range(self.n + 2):
                for c in range(self.n + 2):
                    if r == 0 or r == self.n + 1 or c == 0 or c == self.n + 1:
                        continue
                    elif len(self.possible[r][c]) == 1:
                        num = min(self.possible[r][c])
                        for i in range(1, self.n + 1):
                            if i != r and num in self.possible[i][c]:
                                self.possible[i][c].discard(num)
                                update = True
                            if i != c and num in self.possible[r][i]:
                                self.possible[r][i].discard(num)
                                update = True


    def create_random_board(self):
        nums = np.arange(1, self.n + 1)
        for i in range(1, self.n + 1):
            np.random.shuffle(nums)

            while not Game.satisfy_possible(nums, i, self.possible, self.n):
                np.random.shuffle(nums)
            self.board[i][1:self.n + 1] = nums

    @staticmethod
    def satisfy_possible(nums, i, possible, n):
        return np.all([nums[j] in possible[i][j + 1] for j in range(n)])

    @staticmethod
    def look(arr):
        cur = arr[0]
        res = 1
        for i in range(1, len(arr)):
            if arr[i] > cur:
                cur = arr[i]
                res = res + 1
        return res

    def _check_nums(self):
        return (np.all((self.board[1:self.n + 1, 1:self.n + 1] >=1) & (self.board[1:self.n + 1, 1:self.n + 1] <= self.n)) and
                not np.sum([[np.sum(self.board[r][1:self.n + 1] == num) != 1 for num in range(1, self.n + 1)] for r in range(1, self.n + 1)]) and
                not np.sum([[np.sum(self.board[1:self.n + 1, c] == num) != 1 for num in range(1, self.n + 1)] for c in range(1, self.n + 1)]))

    def looks(self):
        return np.array([[Game.look(self.board[1:self.n + 1, col]) for col in range(1, self.n + 1)],
                         [Game.look(self.board[self.n:0:-1, col]) for col in range(1, self.n + 1)],
                         [Game.look(self.board[row, 1:self.n + 1]) for row in range(1, self.n + 1)],
                         [Game.look(self.board[row, self.n:0:-1]) for row in range(1, self.n + 1)]])

    def _check_restrctions(self):
        return np.all((self.restrictions == 0) | (self.restrictions == self.looks()))

    def valid(self):
        return (self._check_nums() and self._check_restrctions())

    def update_fitness(self):
        col_score = np.sum([1 / (self.n - len(set(self.board[1:self.n + 1, c])) + 1) for c in range(1, self.n + 1)]) / self.n
        failed_restrictions = np.sum((self.restrictions != 0) & (self.restrictions != self.looks()))
        looks_score = 1 / (failed_restrictions + 1)
        self.fitness = col_score * looks_score

    @staticmethod
    def get_fitness(game):
        return game.fitness

    def __repr__(self):
        res = ""
        for i in range(self.n + 2):
            for j in range(self.n + 2):
                if (i == 0 or j == 0 or i == self.n + 1 or j == self.n + 1) and self.board[i][j] == 0:
                    res = res + " "
                else:
                    if self.is_given[i][j]:
                        res = res + bcolors.FAIL + str(self.board[i][j]) + bcolors.ENDC
                    else:
                        res = res + str(self.board[i][j])
                if j == 0 or j == self.n:
                    res = res + " |"
                res = res + " "
            res = res + "\n"
            if i == 0 or i == self.n:
                for _ in range(self.n + 4):
                    res = res + "- "
                res = res + "\n"
        return res


class Tournament():
    @staticmethod
    def select(population, select_rate):
        p1, p2 = Tournament.select_one(population, select_rate), Tournament.select_one(population, select_rate)
        while p1 == p2:
            p1, p2 = Tournament.select_one(population, select_rate), Tournament.select_one(population, select_rate)
        return p1, p2


    @staticmethod
    def select_one(population, select_rate):
        candidates = population.individuals
        n = len(candidates)
        c1, c2 = candidates[np.random.randint(0, n)], candidates[np.random.randint(0, n)]

        if c1.fitness > c2.fitness:
            winner, loser = c1, c2
        else:
            winner, loser = c2, c1

        if np.random.uniform(0, 1) <= select_rate:
            return winner
        return loser


class Population():
    def __init__(self, Nc, board):
        self.individuals = [Game(board) for _ in range(Nc)]
        self.sort_by_fitness()
        # print("========\n")
        # print("Generate new population")
        # print("\n========")

    def sort_by_fitness(self):
        self.individuals.sort(reverse=True, key=Game.get_fitness)


class Skyscraper():
    def __init__(self, board):
        self.board = board

    def solve(self):
        Nc = 1000
        Ne = int(0.05 * Nc)
        Ng = 500000
        Nr = int(0.10 * Nc)
    
        CROSSOVER_RATE = 1.0
        SELECT_RATE = 0.85

        phi = 0
        sigma = 1
        sigma_adjust_rate = 0.9
        success_mutate_count = 0
        mutate_rate = 0.06

        population = Population(Nc, self.board)

        stale = 0
        for generation_count in range(Ng):
            best_fitness = population.individuals[0].fitness
            old_records = [population.individuals[i].fitness for i in range(Nr)]
            if best_fitness == 1:
                is_multiple = None
                for i in range(1, Nr):
                    if population.individuals[i].fitness == 1 and np.any(population.individuals[0].board != population.individuals[i].board):
                        is_multiple = population.individuals[i]
                        break
                return population.individuals[0], generation_count, is_multiple 
            # print(f"Generation {generation_count}: {best_fitness = }, {stale = }")
            
            next_individuals = [copy.deepcopy(population.individuals[e]) for e in range(Ne)]
            for _ in range(Ne, Nc, 2):
                p1, p2 = Tournament.select(population, select_rate=SELECT_RATE)

                if np.random.uniform(0, 1) > CROSSOVER_RATE:
                    next_individuals.append(p1)
                    next_individuals.append(p2)
                    continue

                c1, c2 = Crossover.any(p1, p2)

                if np.random.uniform(0, 1) <= mutate_rate:
                    old_c1_fitness = c1.fitness
                    c1 = Mutate.any(c1)
                    success_mutate_count = success_mutate_count + 1
                    if c1.fitness > old_c1_fitness:
                        phi = phi + 1

                if np.random.uniform(0, 1) <= mutate_rate:
                    old_c2_fitness = c2.fitness
                    c2 = Mutate.any(c2)
                    success_mutate_count = success_mutate_count + 1
                    if c2.fitness > old_c2_fitness:
                        phi = phi + 1

                next_individuals.append(c1)
                next_individuals.append(c2)

            population.individuals = next_individuals
            population.sort_by_fitness()

            phi = 0 if success_mutate_count == 0 else phi / success_mutate_count
            sigma = sigma / sigma_adjust_rate if phi > 0.2 else sigma * sigma_adjust_rate
            mutate_rate = abs(np.random.normal(loc=0.0, scale=sigma, size=None))
            success_mutate_count, phi = 0, 0

            stale = stale + 1
            cur_records = [population.individuals[i].fitness for i in range(Nr)]
            if cur_records > old_records:
                stale = 0

            if stale == 100:
                stale = 0
                population = Population(Nc, self.board)
                sigma = 1
                success_mutate_count, phi = 0, 0
                mutate_rate = 0.06

        return None, None, None



if __name__ == "__main__":
    questions = [[[0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 3],
                  [4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 4, 0, 0, 0]],
                 [[0, 0, 3, 1, 0, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 4, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 3],
                  [3, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
                 [[0, 4, 0, 3, 2, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [3, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 3],
                  [2, 0, 0, 0, 0, 0, 0],
                  [0, 0, 3, 0, 3, 3, 0]],
                 [[0, 2, 3, 0, 0, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2],
                  [4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 3, 0, 0, 0]],
                 [[0, 0, 0, 2, 4, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 4],
                  [2, 0, 0, 5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2],
                  [2, 0, 0, 0, 0, 0, 0],
                  [0, 0, 2, 2, 0, 0, 0]],
                 [[0, 0, 0, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 5],
                  [4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 2, 2, 1, 0, 0]],
                 [[0, 0, 3, 1, 0, 0, 0],
                  [3, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 2, 0],
                  [0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 4, 0, 0]],
                 [[0, 4, 0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 3],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 3],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 2, 2, 2, 0, 0, 0]],
                 [[0, 2, 3, 0, 0, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2],
                  [4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 3, 0, 0, 0]],
                 [[0, 0, 0, 4, 0, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [3, 0, 0, 2, 0, 0, 0],
                  [4, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 2, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0, 0]],]

    SIMULATE_TIMES = 5
    
    all_sum = 0
    for question in questions:
        cur_sum = 0
        for _ in range(SIMULATE_TIMES):
            s = Skyscraper(question)
            res, generation_count, is_multiple = s.solve()
            # print(res)
            # print(f"{generation_count = }")
            cur_sum = cur_sum + generation_count
        print(f"cur_avg = {cur_sum / SIMULATE_TIMES}")
        all_sum = all_sum + cur_sum
    all_avg = all_sum / (len(question) * SIMULATE_TIMES) 
    print(f"{all_avg = }")

    # a = Game(questions[0])
    # b = Game(questions[0])

    # print(a)
    # print(f"{a.valid() = }")
    # print(f"{a.fitness = }")
    # print(b)
    # print(f"{b.valid() = }")
    # print(f"{b.fitness = }")

    # # c = Mutate.inverse(a)
    # # print(c)
    # # print(f"{c.valid() = }")
    # # print(f"{c.fitness = }")

    # d, e = Crossover.partially_mapped(a, b)
    # print(d)
    # print(f"{d.valid() = }")
    # print(f"{d.fitness = }")
    # print(e)
    # print(f"{e.valid() = }")
    # print(f"{e.fitness = }")


    

