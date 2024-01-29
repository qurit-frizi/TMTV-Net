import unittest
import corelib


class TestFunctionToTest(unittest.TestCase):
    def test_function_to_test(self):
        r = corelib.todo_function_to_test(1, 2)
        print('Result=', r)
        assert r == 3

if __name__ == '__main__':
    unittest.main()