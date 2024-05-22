#include "pch.h"
#include "framework.h"
#include "MFC_Memo.h"
#include "MFC_MemoDoc.h"
#include "MFC_MemoView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMFCMemoView

IMPLEMENT_DYNCREATE(CMFCMemoView, CScrollView)

BEGIN_MESSAGE_MAP(CMFCMemoView, CScrollView)
    // 표준 인쇄 명령입니다. 필요없음
    ON_COMMAND(ID_FILE_PRINT, &CScrollView::OnFilePrint)
    ON_COMMAND(ID_FILE_PRINT_DIRECT, &CScrollView::OnFilePrint)
    ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CScrollView::OnFilePrintPreview)
    ON_WM_CHAR()// 키보드 이벤트 핸들러 추가
    // *********************************** 메뉴 ***********************************
    ON_COMMAND(ID_FILE_SAVE_2, &CMFCMemoView::OnFileSave2)
    ON_COMMAND(ID_FILE_OPEN, &CMFCMemoView::OnFileOpen)
    ON_COMMAND(ID_FILE_NEW, &CMFCMemoView::OnFileNew)
    ON_COMMAND(ID_FILE_NEW_WINDOW, &CMFCMemoView::OnFileNewWindow)
    // *********************************** 메뉴 ***********************************
    // *********************************** 캐럿 ***********************************
    ON_WM_SETFOCUS()
    ON_WM_KILLFOCUS()
    // *********************************** 캐럿 ***********************************
    // *********************************** 스크롤 ***********************************
    ON_WM_KEYDOWN()
    ON_WM_VSCROLL()
    ON_WM_HSCROLL()
    // *********************************** 스크롤 ***********************************
END_MESSAGE_MAP()

// CMFCMemoView 생성/소멸

CMFCMemoView::CMFCMemoView() noexcept : 
    m_nCurrentRow(0),           // x초기화
m_nCurrentColumn(0),            // y초기화
m_bCaretVisible(false),         // 캐럿

// *********************************** 스크롤 ***********************************
m_nScrollWidth(1000),           // 임의의 너비 설정
m_nScrollHeight(1000),          // 임의의 높이 설정
m_nPageSize(100)                // 한 페이지의 크기 설정
// *********************************** 스크롤 ***********************************

{
    for (int i = 0; i < Row; ++i) {
        for (int j = 0; j < Columns; ++j) {
            m_textArray[i][j] = '\0';
        }
    }
}


CMFCMemoView::~CMFCMemoView()
{
    // *********************************** 캐럿 ***********************************
    if (m_bCaretVisible)
    {
        HideCaret();
        DestroyCaret();
    }
    // *********************************** 캐럿 ***********************************
}

BOOL CMFCMemoView::PreCreateWindow(CREATESTRUCT& cs)
{
    // TODO: CREATESTRUCT cs를 수정하여 여기에서
    //  Window 클래스 또는 스타일을 수정합니다.

    return CScrollView::PreCreateWindow(cs);
}

// *********************************** 캐럿 ***********************************
void CMFCMemoView::OnInitialUpdate()
{
    CScrollView::OnInitialUpdate();

    // 스크롤 바를 설정합니다.
    SetScrollSizes(MM_TEXT, CSize(m_nScrollWidth, m_nScrollHeight));
}
// *********************************** 캐럿 ***********************************

// CMFCMemoView 그리기

void CMFCMemoView::OnDraw(CDC* pDC)
{
    CMFCMemoDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);
    if (!pDoc)
        return;

    // 폰트 설정 (예시로 Arial, 크기 12 사용)
    CFont font;
    font.CreateFontW(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Arial");
    pDC->SelectObject(&font);

    // 텍스트를 화면에 출력합니다.
    for (int i = 0; i < Row; ++i) 
    {
        CString textString(m_textArray[i]); // wchar_t 배열을 CString으로 변환
        pDC->TextOut(0, i * 20, textString); // CString을 사용하여 출력
    }
}

// *********************************** 스크롤 ***********************************
void CMFCMemoView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
    // 상하좌우 이동에 따라 스크롤 조정
    CSize sizeTotal = GetTotalSize();

    CRect clientRect;
    GetClientRect(&clientRect);
    CSize sizePage(clientRect.Width(), clientRect.Height());

    CPoint ptOrg = GetScrollPosition();

    switch (nChar)
    {
    case VK_UP:
        ptOrg.y = max(ptOrg.y - 20, 0);
        break;
    case VK_DOWN:
        ptOrg.y = min(ptOrg.y + 20, sizeTotal.cy - sizePage.cy);
        break;
    case VK_LEFT:
        ptOrg.x = max(ptOrg.x - 20, 0);
        break;
    case VK_RIGHT:
        ptOrg.x = min(ptOrg.x + 20, sizeTotal.cx - sizePage.cx);
        break;
    }

    ScrollToPosition(ptOrg);

    CScrollView::OnKeyDown(nChar, nRepCnt, nFlags);
}

void CMFCMemoView::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    // 수직 스크롤 동작 처리
    int scrollPos = GetScrollPosition().y;
    int maxPos = GetScrollLimit(SB_VERT);

    switch (nSBCode)
    {
    case SB_LINEUP:
        scrollPos = max(scrollPos - 20, 0);
        break;
    case SB_LINEDOWN:
        scrollPos = min(scrollPos + 20, maxPos);
        break;
    case SB_PAGEUP:
        scrollPos = max(scrollPos - 100, 0);
        break;
    case SB_PAGEDOWN:
        scrollPos = min(scrollPos + 100, maxPos);
        break;
    case SB_THUMBPOSITION:
    case SB_THUMBTRACK:
        scrollPos = nPos;
        break;
    }

    ScrollToPosition(CPoint(GetScrollPosition().x, scrollPos));
    CScrollView::OnVScroll(nSBCode, nPos, pScrollBar);
}

void CMFCMemoView::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    // 수평 스크롤 동작 처리
    int scrollPos = GetScrollPosition().x;
    int maxPos = GetScrollLimit(SB_HORZ);

    switch (nSBCode)
    {
    case SB_LINELEFT:
        scrollPos = max(scrollPos - 20, 0);
        break;
    case SB_LINERIGHT:
        scrollPos = min(scrollPos + 20, maxPos);
        break;
    case SB_PAGELEFT:
        scrollPos = max(scrollPos - 100, 0);
        break;
    case SB_PAGERIGHT:
        scrollPos = min(scrollPos + 100, maxPos);
        break;
    case SB_THUMBPOSITION:
    case SB_THUMBTRACK:
        scrollPos = nPos;
        break;
    }

    ScrollToPosition(CPoint(scrollPos, GetScrollPosition().y));
    CScrollView::OnHScroll(nSBCode, nPos, pScrollBar);
}
// *********************************** 스크롤 ***********************************

// CMFCMemoView 인쇄

BOOL CMFCMemoView::OnPreparePrinting(CPrintInfo* pInfo)
{
    // 기본적인 준비
    return DoPreparePrinting(pInfo);
}

void CMFCMemoView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
    // TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void CMFCMemoView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
    // TODO: 인쇄 후 정리 작업을 추가합니다.
}


// CMFCMemoView 진단

#ifdef _DEBUG
void CMFCMemoView::AssertValid() const
{
    CView::AssertValid();
}

void CMFCMemoView::Dump(CDumpContext& dc) const
{
    CView::Dump(dc);
}

CMFCMemoDoc* CMFCMemoView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
    ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCMemoDoc)));
    return (CMFCMemoDoc*)m_pDocument;
}
#endif //_DEBUG


// CMFCMemoView 메시지 처리기
void CMFCMemoView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
    // 백스페이스 키를 처리합니다.
    if (nChar == VK_BACK) 
    {
        if (m_nCurrentColumn > 0) 
        {
            // 현재 커서 위치가 첫 번째 열이 아니면 현재 줄에서 이전 열로 이동합니다.
            m_nCurrentColumn--;
        }
        else if (m_nCurrentRow > 0) 
        {
            // 현재 커서 위치가 첫 번째 열이고, 현재 줄이 첫 번째 줄이 아니라면 이전 줄로 이동합니다.
            m_nCurrentRow--;
            m_nCurrentColumn = wcslen(m_textArray[m_nCurrentRow]); // 이전 줄의 마지막 열로 이동합니다.

            // 현재 줄의 문자열 길이를 계산합니다.
            int currentRowLength = wcslen(m_textArray[m_nCurrentRow]);

            // 이전 줄이 있고, 현재 줄이 비어있으면 이전 줄을 지웁니다.
            if (currentRowLength == 0) 
            {
                // 현재 줄을 이전 줄로 복사합니다.
                wcscpy_s(m_textArray[m_nCurrentRow], m_textArray[m_nCurrentRow + 1]);

                // 이전 줄을 비웁니다.
                wmemset(m_textArray[m_nCurrentRow + 1], L'\0', Columns);

                // 커서를 이전 줄의 끝으로 이동합니다.
                m_nCurrentColumn = currentRowLength;
            }
        }
        // 현재 커서 위치가 문자열의 끝이 아니라면 백스페이스 처리
        if (m_textArray[m_nCurrentRow][m_nCurrentColumn] != L'\0') 
        {
            m_textArray[m_nCurrentRow][m_nCurrentColumn] = L'\0';
        }
    }
    else if (nChar == VK_RETURN) 
    {
        m_nCurrentRow++;
        m_nCurrentColumn = 0;
    }
    else if (nChar == VK_TAB) 
    {
        // 탭 문자 처리 (탭 공백으로 대체)
        m_textArray[m_nCurrentRow][m_nCurrentColumn] = L' ';
        m_nCurrentColumn++;
    }
    else if (nChar == VK_ESCAPE)
    {
    }
    else {
        // 일반 문자 처리
        m_textArray[m_nCurrentRow][m_nCurrentColumn] = (wchar_t)nChar;
        m_nCurrentColumn++;
    }

    // 뷰를 다시 그려서 텍스트와 캐럿을 표시합니다.
    UpdateCaretPosition();
    Invalidate();

    // 부모 클래스의 OnChar 함수를 호출하여 기본 키보드 입력 처리를 수행합니다.
    CView::OnChar(nChar, nRepCnt, nFlags);
}


void CMFCMemoView::OnFileSave2()
{
    // 파일 저장 대화상자 생성
    CFileDialog fileDlg(FALSE, _T("txt"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Text Files (*.txt)|*.txt||"), this);

    // 파일 저장 대화상자 열기
    if (fileDlg.DoModal() == IDOK)
    {
        CString pathName = fileDlg.GetPathName();

        // 파일 열기
        CFile file(pathName, CFile::modeCreate | CFile::modeWrite);
        CArchive ar(&file, CArchive::store);

        // 텍스트 데이터를 파일에 저장
        for (int i = 0; i < Row; ++i)
        {
            ar.WriteString(CString(m_textArray[i]) + _T("\n"));
        }

        ar.Close();
        file.Close();
    }
}

void CMFCMemoView::OnFileOpen()
{
    // 파일 열기 대화상자 생성
    CFileDialog fileDlg(TRUE, _T("txt"), NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, _T("Text Files (*.txt)|*.txt||"), this);

    // 파일 열기 대화상자 열기
    if (fileDlg.DoModal() == IDOK)
    {
        CString pathName = fileDlg.GetPathName();

        // 파일 열기
        CFile file(pathName, CFile::modeRead);
        CArchive ar(&file, CArchive::load);

        // 텍스트 데이터 초기화
        for (int i = 0; i < Row; ++i)
        {
            memset(m_textArray[i], '\0', Columns);
        }

        // 파일에서 텍스트 데이터 읽기
        int row = 0;
        CString line;
        while (ar.ReadString(line))
        {
            if (row < Row)
            {
                wcscpy_s(m_textArray[row], Columns, line.GetBuffer());
                line.ReleaseBuffer();
                row++;
            }
        }

        ar.Close();
        file.Close();

        // 텍스트 데이터 출력
        Invalidate();
    }
}

void CMFCMemoView::OnFileNew()
{
    // 텍스트 데이터 초기화
    for (int i = 0; i < Row; ++i)
    {
        memset(m_textArray[i], '\0', Columns);
    }

    // 커서 위치 초기화
    m_nCurrentRow = 0;
    m_nCurrentColumn = 0;

    // 텍스트 데이터 출력
    Invalidate();
}

void CMFCMemoView::OnFileNewWindow()
{
    // 새 윈도우 생성
    CFrameWnd* pNewFrame = new CFrameWnd;
    pNewFrame->Create(NULL, _T("New Window"), WS_OVERLAPPEDWINDOW);
    pNewFrame->ShowWindow(SW_SHOW);
}

// *********************************** 캐럿 ***********************************
void CMFCMemoView::OnSetFocus(CWnd* pOldWnd)
{
    CView::OnSetFocus(pOldWnd);
    CreateCaretIfNeeded();
    ShowCaret();
    m_bCaretVisible = true;
}

void CMFCMemoView::OnKillFocus(CWnd* pNewWnd)
{
    CView::OnKillFocus(pNewWnd);
    HideCaret();
    DestroyCaret();
    m_bCaretVisible = false;
}

void CMFCMemoView::CreateCaretIfNeeded()
{
    if (!m_bCaretVisible)
    {
        CreateSolidCaret(1, 20); // 폭: 2, 높이: 20 (필요에 따라 조정)
        ShowCaret();
        m_bCaretVisible = true;
    }
}

void CMFCMemoView::UpdateCaretPosition()
{
    CClientDC dc(this);
    CFont font;
    font.CreateFontW(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Arial");
    CFont* pOldFont = dc.SelectObject(&font);

    int x = dc.GetTextExtent(CString(m_textArray[m_nCurrentRow], m_nCurrentColumn)).cx;
    int y = m_nCurrentRow * 20;

    SetCaretPos(CPoint(x, y));

    dc.SelectObject(pOldFont);
}
// *********************************** 캐럿 ***********************************