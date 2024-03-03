/*
1. 텍스트 입력 받을 수 있어야함. >> 1차원 배열 ㄴㄴ 2차원 배열로 메모리는 동적할당 받아서 해야함.
2. 메모장 상단에 메뉴바(툴바)가 있어야함
2-1. 메뉴바는 파일, 편집, 서식으로 구성 되어야함.
+ 새로 만들기, 새 창 열기, 저장, 열기 ...
2-2. 파일, 편집, 서식에는 각 카테고리에 맞는 기능이 있어야함.
3. 키보드 기능(Home, End, Delete, Insert, Page Up, Page Down, Tab, Back_Space, Enter, Up,Down,left,Right)을 인식 해야함.
4. 메모장 스크롤 기능이 있어야함. >> 윈도우 창 크기에 맞게 조절이 되어야함.
5. 캐럿이 존재해야함. 
6.     SetScrollRange(hWnd, SB_VERT, 0, MAX_ROWS - 1, TRUE);
    SetScrollRange(hWnd, SB_HORZ, 0, MAX_COLUMNS - 1, TRUE);
*** 사용한 메모리는 반납될 수 있도록 해야함 (메모리 누수 방지)

문제점
ㅗ Enter(줄 바꿈) 전까지 현재 입력 중인 줄의 텍스트가 안보임
개 지랄 맞은 경우 창에 글자를 작성 후 줄 바꾼 후 글자를 다 지우고 다시 작성하면 글자가 작성되면서 보임 ㅅㅂ
ㅗ 새 창 열기 후 새 창을 닫고 기존 창으로 넘어가면 새 창의 데이터가 남음
ㅗ 메세지 창 처리 안함
ㅗ 캐럿이 아직도 병50임 영어만 쓰면 맞는데 여기다 띄어쓰기랑 한글 이런거 더하면 캐럿이 문장의 끝을 못따라감.
  >> 글꼴 수정으로 영문의 폭은 찾아서 해결함 (띄어쓰기 포함) but 한글은 아직 안됌.
ㅗ 열기로 파일을 열고 해당 파일 데이터 수정이 불가 ㅅㅂ
*/

/* 구현 완료
1. 한글과 영어 입력가능
2. 메뉴바 생성과 기능들 내역 출력 가능
새로 만들기, 새 창 열기, 저장 및 열기 가능
3. 캐럿 출력 (영어는 고정폭을 갖는 글꼴을 찾아서 캐럿의 위치가 맞지만 한글은 맞지 않음)
4. HOME, END, DELETE BACK SPACE, ENTER, TAB 방향키키 먹음 INSERT와 PAGE UP, DOWN 안됌
5.스크롤바 출력 가능 (구현은 안됌)
*/
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <imm.h>
#include "head.h" // 리소스 헤더 포함

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

HINSTANCE hInst;
HWND hEdit;
WCHAR **textBuffer = NULL;		// 2차원 배열로 변경
int textRows = 0;
int textColumns = 0;
int caretRow = 0;
int caretColumn = 0;
const int MAX_ROWS = 10000;		// 최대 행 수
const int MAX_COLUMNS = 10000;	// 최대 열 수
const int TEXT_HEIGHT = 16;
const int PAGE_SIZE = 10;		// 한번에 스크롤한 페이지
int tabWidth = 4;

// 새 창을 생성하는 함수
HWND CreateNewMemoWindow(HINSTANCE hInstance) {
    textRows = 0;		// 텍스트 행 수 초기화
    textColumns = 0;	// 텍스트 열 수 초기화
    caretRow = 0;		// 커서 행 위치 초기화
    caretColumn = 0;	// 커서 열 위치 초기화
    WNDCLASS wc = {0};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("MyMemoClass");

    RegisterClass(&wc);

    HWND hWnd = CreateWindow(wc.lpszClassName, TEXT("동오의 비밀의 새 창 메모장"), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL);
    // "중요" 새 창을 화면에 표시함.
    ShowWindow(hWnd, SW_SHOWNORMAL); 

    // 텍스트 입력 받는거.
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return hWnd;
}

// WinMain 함수
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    MSG msg;
    WNDCLASS wc = {0};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("MyMemoClass");

    if (!RegisterClass(&wc)) return 0;

    HWND hWnd = CreateWindow(
        wc.lpszClassName,									// 윈도우 클래스 이름
        TEXT("동오의 비밀의 메모장"),						// 윈도우 제목
        WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_HSCROLL,		// 윈도우 스타일에 수직 및 수평 스크롤바 추가 기능 추가해야지 구현됌
        CW_USEDEFAULT,										// x 위치
        CW_USEDEFAULT,										// y 위치
        640,												// 너비
        480,												// 높이
        NULL,												// 부모 윈도우 핸들
        LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)),	// 메뉴 핸들
        hInstance,											// 인스턴스 핸들
        NULL												// 추가 매개변수
    );

    if (!hWnd) return 0;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // 텍스트 버퍼를 할당
    textBuffer = (WCHAR**)malloc(MAX_ROWS * sizeof(WCHAR*));
    if (textBuffer == NULL) {
        MessageBox(NULL, TEXT("메모리 할당에 실패했습니다."), TEXT("에러"), MB_OK | MB_ICONERROR);
        return 0;
    }
    for (int i = 0; i < MAX_ROWS; ++i) {
        textBuffer[i] = (WCHAR*)malloc(MAX_COLUMNS * sizeof(WCHAR));
        if (textBuffer[i] == NULL) {
            MessageBox(NULL, TEXT("메모리 할당에 실패했습니다."), TEXT("에러"), MB_OK | MB_ICONERROR);
            return 0;
        }
        textBuffer[i][0] = L'\0'; // 각 행의 첫 번째 열에 NULL 문자 삽입
    }

    // 텍스트 입력 받는거.
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 할당된 메모리 해제
    for (int i = 0; i < MAX_ROWS; ++i) {
        free(textBuffer[i]);
    }
    free(textBuffer);

    return (int)msg.wParam;
}

// 윈도우 프로시저
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CREATE:
            CreateCaret(hWnd, NULL, 1, TEXT_HEIGHT);
            ShowCaret(hWnd);
            SetCaretPos(0, 0);
            caretRow = 0;
            caretColumn = 0;
			SetScrollRange(hWnd, SB_VERT, 0, MAX_ROWS - 1, TRUE);
			SetScrollRange(hWnd, SB_HORZ, 0, MAX_COLUMNS - 1, TRUE);
            break;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case ID_FILE_NEW:
                    // 텍스트 초기화
                    for (int i = 0; i < MAX_ROWS; ++i) {
                        textBuffer[i][0] = L'\0';
                    }
                    textRows = 0;
                    textColumns = 0;
                    caretRow = 0;
                    caretColumn = 0;
                    SetCaretPos(0, 0);
                    InvalidateRect(hWnd, NULL, TRUE);
                    break;

                case ID_FILE_NEW_WINDOW:
                    // 새 창 열기
                    CreateNewMemoWindow(hInst); 
                    break;

				case ID_FILE_SAVE:
					{
						OPENFILENAME save;
						WCHAR szFileName[MAX_PATH] = L"";

						ZeroMemory(&save, sizeof(save));
    
						save.lStructSize = sizeof(save);
						save.hwndOwner = hWnd;
						save.lpstrFilter = L"텍스트 파일 (*.txt)\0*.TXT\0모든 파일 (*.*)\0*.*\0";
						save.lpstrFile = szFileName;
						save.lpstrFile[0] = L'\0';    
						save.nMaxFile = sizeof(szFileName) / sizeof(*szFileName);
						save.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    
						if (GetSaveFileName(&save) == TRUE) {
							FILE* file = _wfopen(save.lpstrFile, L"w"); // 파일 열기
							if (file != NULL) {
								// 모든 행에 대해 파일에 쓰기
								for (int i = 0; i < textRows; ++i) {
									fwprintf(file, L"%s\n", textBuffer[i]);
								}
								fclose(file); // 파일 닫기
							}
						}
					}
					break;

				case ID_FILE_OPEN:
					{
						OPENFILENAME ofen;
						WCHAR szFileName[MAX_PATH] = L"";

						ZeroMemory(&ofen, sizeof(ofen));
						ofen.lStructSize = sizeof(ofen);
						ofen.hwndOwner = hWnd;
						ofen.lpstrFilter = L"텍스트 파일 (*.txt)\0*.TXT\0모든 파일 (*.*)\0*.*\0";
						ofen.lpstrFile = szFileName;
						ofen.lpstrFile[0] = L'\0';
						ofen.nMaxFile = sizeof(szFileName) / sizeof(*szFileName);
						ofen.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

						if (GetOpenFileName(&ofen) == TRUE) {
							FILE* file = _wfopen(ofen.lpstrFile, L"r"); // 파일 열기
							if (file != NULL) {
								// 텍스트를 저장할 임시 버퍼 할당
								WCHAR tempBuffer[MAX_COLUMNS];
								while (fgetws(tempBuffer, MAX_COLUMNS, file) != NULL) {
									tempBuffer[wcslen(tempBuffer) - 1] = L'\0'; // 개행 문자 제거
									wcscpy_s(textBuffer[textRows], MAX_COLUMNS, tempBuffer);
									textRows++;
								}
								fclose(file); // 파일 닫기
            
								// 캐럿 생성
								CreateCaret(hWnd, NULL, 1, TEXT_HEIGHT);     
								ShowCaret(hWnd);
								SetCaretPos(0, 0);
								caretRow = 0;
								caretColumn = 0;
								InvalidateRect(hWnd, NULL, TRUE); // 윈도우를 다시 그리도록 강제        
							}   
						}
					}
					break;
					
					case WM_VSCROLL:
					{
						int scrollY = GetScrollPos(hWnd, SB_VERT);
						switch (LOWORD(wParam)) {
						case SB_LINEUP:   // 위로 한 줄 스크롤
							scrollY--;
							break;
        
						case SB_LINEDOWN: // 아래로 한 줄 스크롤
							scrollY++;
							break;
        
						case SB_PAGEUP:   // 위로 한 페이지 스크롤
							scrollY -= PAGE_SIZE;
							break;
        
						case SB_PAGEDOWN: // 아래로 한 페이지 스크롤
							scrollY += PAGE_SIZE;
							break;
        
						case SB_THUMBPOSITION: // 스크롤바 드래그
							scrollY = HIWORD(wParam);
							break;
						}
    
						// 스크롤 위치가 범위를 벗어나지 않도록 조정
						scrollY = max(0, min(MAX_ROWS - 1, scrollY));
    
						// 스크롤 위치를 설정
						SetScrollPos(hWnd, SB_VERT, scrollY, TRUE);
    
						// 화면을 다시 그림
						InvalidateRect(hWnd, NULL, TRUE);
						break;
					}

				case WM_HSCROLL:
				{	
					int scrollX = GetScrollPos(hWnd, SB_HORZ);
					switch (LOWORD(wParam)) {
					case SB_LINELEFT:   // 왼쪽으로 한 칸 스크롤
						scrollX--;
						break;
					case SB_LINERIGHT: // 오른쪽으로 한 칸 스크롤
						scrollX++;
						break;
					case SB_PAGELEFT:   // 왼쪽으로 한 페이지 스크롤
						scrollX -= PAGE_SIZE;
						break;
					case SB_PAGERIGHT: // 오른쪽으로 한 페이지 스크롤
						scrollX += PAGE_SIZE;
						break;
					case SB_THUMBPOSITION: // 스크롤바 드래그
						scrollX = HIWORD(wParam);
						break;
					}
    
					// 스크롤 위치가 범위를 벗어나지 않도록 조정
					scrollX = max(0, min(MAX_COLUMNS - 1, scrollX));
    
					// 스크롤 위치를 설정
					SetScrollPos(hWnd, SB_HORZ, scrollX, TRUE);
    
					// 화면을 다시 그림
					InvalidateRect(hWnd, NULL, TRUE);
					break;
				}

                case ID_FILE_EXIT:
                    DestroyWindow(hWnd);
                    break;
            }
            break;

		case WM_SIZE:
            break;

        case WM_KEYDOWN:
            switch (wParam) {
                case VK_HOME:
                    caretColumn = 0;
                    break;

                case VK_END:
                    caretColumn = textColumns;
                    break;

                case VK_LEFT:
                    if (caretColumn > 0) caretColumn--;
                    break;

                case VK_RIGHT:
                    if (caretColumn < textColumns) caretColumn++;
                    break;

                case VK_UP:
                    if (caretRow > 0) caretRow--;
                    break;

                case VK_DOWN:
                    if (caretRow < textRows) caretRow++;
                    break;

                case VK_TAB:
                    // 탭 문자 대신 스페이스를 넣어 일정 간격을 벌려줍니다.
                    for (int i = textColumns; i > caretColumn; i--) {
                        textBuffer[caretRow][i + tabWidth] = textBuffer[caretRow][i];
                    }
                    // 일정 간격만큼 스페이스를 넣어줍니다.
                    for (int i = 0; i < tabWidth; ++i) {
                        textBuffer[caretRow][caretColumn + i] = L' ';
                    }
                    caretColumn += tabWidth;
                    textColumns += tabWidth;
                    break;

                case VK_RETURN:
                    // 엔터를 누르면 새로운 줄로 이동하도록함
                    for (int i = textRows; i > caretRow; i--) {
                        wcscpy_s(textBuffer[i], MAX_COLUMNS, textBuffer[i - 1]);
                    }
                    textRows++;
                    caretRow++;
                    textBuffer[caretRow][0] = L'\0'; // 새로운 줄의 첫 번째 열에 NULL 문자 삽입
                    caretColumn = 0;
                    break;

				case VK_BACK:
					// 백스페이스를 누르면 커서의 왼쪽에 있는 문자를 삭제합니다.
					if (caretColumn > 0) {
						for (int i = caretColumn - 1; i < textColumns; i++) {
							textBuffer[caretRow][i] = textBuffer[caretRow][i + 1];
						}
						caretColumn--;
						textColumns--;
					} else if (caretRow > 0) {  // 줄 바꿈을 고려합니다.
						caretRow--; //Row를 하나 줄이고 그 줄 끝에서 왼쪽으로 하나씩 삭제하도록
						caretColumn = wcslen(textBuffer[caretRow]);  // 이전 줄의 길이로 caretColumn을 설정합니다.
						textColumns = caretColumn;
					}
					break;

                case VK_DELETE:
                    // 딜리트를 누르면 커서의 위치에 있는 문자를 삭제합니다.
                    if (caretColumn < textColumns) {
                        for (int i = caretColumn; i < textColumns; i++) {
                            textBuffer[caretRow][i] = textBuffer[caretRow][i + 1];
                        }
                        textColumns--;
                    }
                    break;
            }

            SetCaretPos(8 * caretColumn, TEXT_HEIGHT * caretRow); //8은 폰트 너비 이걸 조절?
            InvalidateRect(hWnd, NULL, TRUE);
            break;


		case WM_CHAR:
			if (textRows < MAX_ROWS && textColumns < MAX_COLUMNS) {
				if ((wParam >= 0x20 && wParam <= 0x7E) ||   // ASCII 문자 범위 이게 영어
					(wParam >= 0xAC00 && wParam <= 0xD7AF)) {  // 한글 유니코드 범위
            
						// 버퍼 오버플로우 방지
						if (textColumns < MAX_COLUMNS - 1) {
                
							// 삽입할 문자 이후의 문자들을 오른쪽으로 이동
							for (int i = textColumns; i > caretColumn; i--) {
								textBuffer[caretRow][i] = textBuffer[caretRow][i - 1];
							}
                
							// 입력된 문자를 삽입
							textBuffer[caretRow][caretColumn] = wParam;
							caretColumn++;
							textColumns++;
						}
				}
			}
    
			// NULL 문자를 추가 << 이 줄 넣으니 첫 줄에 이상한 단어 들어가는 현상 사라짐
			// TEXTOut() 함수는 출력할 문자와 그 문자열으 길이를 인자로 받음 하지만 이때 문자열의 길이를 정학하게 알아야함.
			// 그래서 NULL을 통해 끝을 맺어줘야 길이를 알 수 있기에 NULL값 을 넣어주는 코드로 해결
			textBuffer[caretRow][textColumns] = '\0';
    
			SetCaretPos(8 * caretColumn, TEXT_HEIGHT * caretRow);

			InvalidateRect(hWnd, NULL, TRUE);
			break;


        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            SetTextColor(hdc, RGB(0, 0, 0));
            SetBkColor(hdc, RGB(255, 255, 255));

			HFONT hFont = CreateFont(17, 0, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS,
                          CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FIXED_PITCH | FF_MODERN, TEXT("Courier New"));
			HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);
			//FF_MODERN는 고정폭 글꼴을 의미하는 플래그이며, Courier New는 고정폭 글꼴의 한 코드임.
			//문제점 Courier New << 이친구는 영문만 고정폭 지원

            // 모든 텍스트를 출력
            for (int i = 0; i < textRows; ++i) {
                TextOut(hdc, 0, TEXT_HEIGHT * i, textBuffer[i], wcslen(textBuffer[i]));
            }
			SelectObject(hdc, hOldFont);
			DeleteObject(hFont);

            EndPaint(hWnd, &ps);
            break;
        }

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

